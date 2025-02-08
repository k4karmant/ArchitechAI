import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 🔹 Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 🔹 Load Pretrained U-Net as Generator
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            nn.MaxPool2d(2),
            conv_block(256, 512),
            nn.MaxPool2d(2),
        )

        self.bottleneck = conv_block(512, 1024)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            conv_block(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            conv_block(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # No Sigmoid() (handled in loss)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# 🔹 Define Discriminator (CNN-based)
class Discriminator(nn.Module):
    def __init__(self, in_channels=5):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # Outputs probability
        )

    def forward(self, x):
        return self.model(x)

# 🔹 Load Dataset
class FloorPlanDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data = pickle.load(f)

        if len(data) < 4:
            return None, None

        inside_mask, boundary_mask, interiorWall_mask, interiordoor_mask = data[:4]
        real_floorplan = boundary_mask  # Ground truth

        inside_mask = torch.tensor(inside_mask, dtype=torch.float32).unsqueeze(0)  
        boundary_mask = torch.tensor(boundary_mask, dtype=torch.float32).unsqueeze(0)
        interiorWall_mask = torch.tensor(interiorWall_mask, dtype=torch.float32).unsqueeze(0)
        interiordoor_mask = torch.tensor(interiordoor_mask, dtype=torch.float32).unsqueeze(0)

        input_tensor = torch.cat([inside_mask, boundary_mask, interiorWall_mask, interiordoor_mask], dim=0)
        output_tensor = torch.tensor(real_floorplan, dtype=torch.float32).unsqueeze(0)  

        return input_tensor, output_tensor

# 🔹 Train GAN
def train_gan(generator, discriminator, train_loader, val_loader, epochs=10):
    criterion_GAN = nn.BCEWithLogitsLoss()  # For Discriminator
    criterion_L1 = nn.L1Loss()  # L1 Loss for U-Net generator
    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

    # 🔹 Learning Rate Scheduler (Reduces LR over time)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)

    scaler = GradScaler()  # ✅ Mixed Precision (AMP)
    best_val_loss = float("inf")
    early_stopping_counter = 0
    accumulation_steps = 4  # ✅ Gradient Accumulation

    # ✅ Pretrain Discriminator for Stability
    print("🔄 Pretraining Discriminator for 2 epochs...")
    for _ in range(2):
        for inputs, real_images in train_loader:
            if inputs is None:
                continue
            inputs, real_images = inputs.to(device), real_images.to(device)
            fake_images = generator(inputs).detach()
            fake_images_resized = F.interpolate(fake_images, size=(256, 256), mode="bilinear", align_corners=False)
            real_preds = discriminator(torch.cat([inputs, real_images], dim=1))
            fake_preds = discriminator(torch.cat([inputs, fake_images_resized], dim=1))
            loss_real = criterion_GAN(real_preds, torch.ones_like(real_preds))
            loss_fake = criterion_GAN(fake_preds, torch.zeros_like(fake_preds))
            loss_D = (loss_real + loss_fake) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
    print("✅ Discriminator Pretraining Complete!")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_loss_G, total_loss_D = 0, 0
        correct_real, correct_fake, total_samples = 0, 0, 0

        tqdm_bar = tqdm(train_loader, desc=f"🛠️ Training Epoch {epoch+1}/{epochs}", unit="batch")

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        for batch_idx, (inputs, real_images) in enumerate(tqdm_bar):
            if inputs is None:
                continue

            inputs, real_images = inputs.to(device), real_images.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            with autocast():  # ✅ Mixed Precision Training
                # 🔹 Generate Fake Images
                fake_images = generator(inputs)
                fake_images_resized = F.interpolate(fake_images, size=(256, 256), mode="bilinear", align_corners=False)

                # 🔹 Train Discriminator
                real_preds = discriminator(torch.cat([inputs, real_images], dim=1))
                fake_preds = discriminator(torch.cat([inputs, fake_images_resized.detach()], dim=1))
                output_shape = real_preds.shape  
                real_labels = torch.ones(output_shape).to(device)
                fake_labels = torch.zeros_like(real_labels)

                loss_real = criterion_GAN(real_preds, real_labels)
                loss_fake = criterion_GAN(fake_preds, fake_labels)
                loss_D = (loss_real + loss_fake) / 2

            scaler.scale(loss_D / accumulation_steps).backward()  # ✅ Gradient Accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_D.zero_grad()

            # 🔹 Train Generator
            with autocast():
                fake_preds = discriminator(torch.cat([inputs, fake_images_resized], dim=1))
                loss_GAN = criterion_GAN(fake_preds, real_labels)
                loss_L1 = criterion_L1(fake_images_resized, real_images) * 100
                loss_G = loss_GAN + loss_L1

            scaler.scale(loss_G / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer_G)
                scaler.update()
                optimizer_G.zero_grad()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

            accuracy = (correct_real + correct_fake) / (2 * total_samples) * 100
            tqdm_bar.set_postfix(loss_G=f"{loss_G.item():.4f}", loss_D=f"{loss_D.item():.4f}", acc=f"{accuracy:.2f}%")

        scheduler_G.step()
        scheduler_D.step()  # ✅ Adjust learning rates dynamically

        print(f"\n📢 Epoch [{epoch+1}/{epochs}] - Loss_G: {total_loss_G:.4f}, Loss_D: {total_loss_D:.4f}, Acc: {accuracy:.2f}%")

        # 🔹 Early Stopping
        if total_loss_G < best_val_loss:
            best_val_loss = total_loss_G
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 3:
            print("🚀 Early stopping triggered! Training stopped.")
            break


# 🔹 Run Training
generator = UNet().to(device)
discriminator = Discriminator().to(device)
train_gan(generator, discriminator, DataLoader(FloorPlanDataset("../../pickle/train"), batch_size=16, shuffle=True), None, epochs=10)
