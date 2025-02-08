import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pickle
import os
import numpy as np
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
        room_node = data[4] if len(data) > 4 else []

        inside_mask = torch.tensor(inside_mask, dtype=torch.float32).unsqueeze(0)  
        boundary_mask = torch.tensor(boundary_mask, dtype=torch.float32).unsqueeze(0)
        interiorWall_mask = torch.tensor(interiorWall_mask, dtype=torch.float32).unsqueeze(0)
        interiordoor_mask = torch.tensor(interiordoor_mask, dtype=torch.float32).unsqueeze(0)

        input_tensor = torch.cat([inside_mask, boundary_mask, interiorWall_mask, interiordoor_mask], dim=0)
        label = room_node[0]['category'] if room_node else 0  

        return input_tensor, label

# 🔹 Define CNN Model (Using Pretrained ResNet-34)
class FloorPlanCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(FloorPlanCNN, self).__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)  # ✅ Use Pretrained Model
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust input channels
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 🔹 Training Function (Optimized with Mixed Precision & Early Stopping)
def train_model(model, train_loader, val_loader, epochs=10):
    best_val_acc = 0
    early_stopping_counter = 0
    scaler = GradScaler()  # ✅ Mixed Precision Training
    accumulation_steps = 2  # ✅ Gradient Accumulation

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        tqdm_bar = tqdm(train_loader, desc=f"🛠️ Training Epoch {epoch+1}/{epochs}", unit="batch")
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(tqdm_bar):
            if inputs is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():  # ✅ Enable Mixed Precision
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps  # ✅ Scale Loss

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:  # ✅ Only update weights every N batches
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            tqdm_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%")

        train_acc = 100 * correct / total
        val_acc = validate_model(model, val_loader)

        print(f"\n📢 Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # 🔹 Early Stopping (Stop if No Improvement in 3 Epochs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 3:
            print("🚀 Early stopping triggered! Training stopped.")
            break

# 🔹 Validation Function (Optimized)
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        tqdm_bar = tqdm(val_loader, desc="🔎 Validating", unit="batch")
        for inputs, labels in tqdm_bar:
            if inputs is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# 🔹 Run Training in Safe Mode for Windows (Fix Multiprocessing Issue)
if __name__ == "__main__":
    train_dataset = FloorPlanDataset("C:/Users/user/Desktop/New folder/Architech AI/pickle/train")
    val_dataset = FloorPlanDataset("C:/Users/user/Desktop/New folder/Architech AI/pickle/val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # ✅ Larger Batch Size (64)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = FloorPlanCNN(num_classes=12).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, epochs=10)

    torch.save(model.state_dict(), "floorplan_cnn.pth")
    print("✅ Model saved as floorplan_cnn.pth")
