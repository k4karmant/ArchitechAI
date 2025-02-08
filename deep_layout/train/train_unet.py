import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # ✅ Mixed Precision Training
from tqdm import tqdm  # ✅ Progress Bar

# 🔹 Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 🔹 Define Custom Dataset to Load Pickle Files
class FloorPlanDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data = pickle.load(f)

        if len(data) < 4:  # ✅ Skip corrupted files
            print(f"⚠️ Warning: Corrupt pickle file {self.file_list[idx]} (Got {len(data)} elements)")
            return None, None

        inside_mask, boundary_mask, interiorWall_mask, interiordoor_mask = data[:4]
        output_mask = boundary_mask  # ✅ Use boundary_mask as the target output

        inside_mask = torch.tensor(inside_mask, dtype=torch.float32).unsqueeze(0)  
        boundary_mask = torch.tensor(boundary_mask, dtype=torch.float32).unsqueeze(0)
        interiorWall_mask = torch.tensor(interiorWall_mask, dtype=torch.float32).unsqueeze(0)
        interiordoor_mask = torch.tensor(interiordoor_mask, dtype=torch.float32).unsqueeze(0)

        input_tensor = torch.cat([inside_mask, boundary_mask, interiorWall_mask, interiordoor_mask], dim=0)
        output_tensor = torch.tensor(output_mask, dtype=torch.float32).unsqueeze(0)  

        return input_tensor, output_tensor

# 🔹 Define Optimized U-Net Model
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
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # ✅ REMOVE Sigmoid()
        )
    def forward(self, x):
            x = self.encoder(x)
            x = self.bottleneck(x)
            x = self.decoder(x)
            return x

# 🔹 Optimized Training Function (AMP, Early Stopping, Gradient Accumulation)
def train_model(model, train_loader, val_loader, epochs=10):
    best_val_loss = float("inf")
    early_stopping_counter = 0
    scaler = GradScaler()
    accumulation_steps = 4  # ✅ Accumulate gradients to use larger batch sizes

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        tqdm_bar = tqdm(train_loader, desc=f"🛠️ Training Epoch {epoch+1}/{epochs}", unit="batch")
        optimizer.zero_grad()

        for i, (inputs, targets) in enumerate(tqdm_bar):
            if inputs is None:
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                
                # ✅ Resize U-Net output to match target shape
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode="bilinear", align_corners=False)

                loss = criterion(outputs, targets) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:  # ✅ Update weights only after accumulation steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = validate_model(model, val_loader)
        print(f"\n📢 Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 3:
            print("🚀 Early stopping triggered! Training stopped.")
            break

# 🔹 Optimized Validation Function
def validate_model(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        tqdm_bar = tqdm(val_loader, desc="🔎 Validating", unit="batch")
        for inputs, targets in tqdm_bar:
            if inputs is None:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)

# 🔹 Run Training in Safe Mode for Windows (Fix Multiprocessing Issue)
if __name__ == "__main__":
    train_dataset = FloorPlanDataset("C:/Users/user/Desktop/New folder/Architech AI/pickle/train")
    val_dataset = FloorPlanDataset("C:/Users/user/Desktop/New folder/Architech AI/pickle/val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # ✅ Larger Batch Size
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # ✅ FIXED
  # ✅ Binary Cross Entropy for segmentation
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # ✅ Faster convergence

    train_model(model, train_loader, val_loader, epochs=10)

    torch.save(model.state_dict(), "unet_floorplan.pth")
    print("✅ Model saved as unet_floorplan.pth")
