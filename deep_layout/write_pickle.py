import torch
import numpy as np
import os
import pickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(path):
    with Image.open(path) as temp:
        image_array = np.asarray(temp, dtype=np.uint8)

    image_tensor = torch.tensor(image_array, dtype=torch.uint8).cuda()

    boundary_mask = image_tensor[:, :, 0].cpu().numpy()
    category_mask = image_tensor[:, :, 1].cpu().numpy()
    index_mask = image_tensor[:, :, 2].cpu().numpy()
    inside_mask = image_tensor[:, :, 3].cpu().numpy()

    pkl_path = path.replace('dataset/val', 'pickle/val').replace('png', 'pkl')
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump([inside_mask, boundary_mask, category_mask, index_mask], pkl_file)

def batch_process_images(image_paths):
    with ThreadPoolExecutor(max_workers=8) as executor:  # Use 8 parallel threads
        executor.map(process_image, image_paths)

if __name__ == '__main__':
    dataset_dir = "dataset/val"
    image_paths = [os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir)]

    print(f"Processing {len(image_paths)} images...")
    batch_process_images(image_paths)
    print("Conversion completed!")
