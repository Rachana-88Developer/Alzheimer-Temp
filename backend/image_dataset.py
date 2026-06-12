"""
OASIS Image Dataset Loader
Downloads from: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
"""
import os
# Force Kaggle to download to D drive project folder
os.environ["KAGGLEHUB_CACHE"] = r"d:\IEEE_ML\cache"


import shutil

KAGGLE_URL = "ninadaithal/imagesoasis"

def download_image_dataset():
    print("=" * 60)
    print("  DOWNLOADING OASIS MRI IMAGE DATASET FROM KAGGLE")
    print("=" * 60)
    
    # Download dataset
    path = kagglehub.dataset_download(KAGGLE_URL)
    print(f"[OK] Image dataset downloaded to: {path}")
    
    # Check directory structure
    # Usually: Data/Non-Demented, Data/Mild-Demented, etc.
    data_dir = os.path.join(path, "Data")
    if not os.path.exists(data_dir):
        # Sometimes it's directly in the root or a different subfolder
        data_dir = path
        
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"[INFO] Found {len(classes)} classes: {classes}")
    
    # Count images
    total_images = sum([len(files) for r, d, files in os.walk(data_dir)])
    print(f"[INFO] Total MRI images found: {total_images}")
    
    return data_dir, classes

if __name__ == "__main__":
    download_image_dataset()
