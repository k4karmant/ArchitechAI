import os
import shutil
import random

# Set dataset paths
source_folder = "dataset/floorplan_dataset"  # Change this to your dataset folder
train_folder = "dataset/train"
val_folder = "dataset/val"

# Create directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get list of all files
all_files = os.listdir(source_folder)
random.shuffle(all_files)  # Shuffle for randomness

# Define split ratio
split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(all_files) * split_ratio)

# Split dataset
train_files = all_files[:split_index]
val_files = all_files[split_index:]

# Move files to respective directories
for file in train_files:
    shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))

for file in val_files:
    shutil.move(os.path.join(source_folder, file), os.path.join(val_folder, file))

print(f"Dataset split complete! {len(train_files)} training files, {len(val_files)} validation files.")