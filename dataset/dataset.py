import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class CatsDogsDataset(Dataset):
    """
    Custom Dataset for loading Cat and Dog images.
    """

    def __init__(self, file_paths, labels, img_size=(64, 64)):
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # (C,H,W)

        return img_tensor, torch.tensor(label, dtype=torch.long)


def create_dataset(cat_dir, dog_dir):
    """
    Loads file paths and labels for train/test split.
    """
    cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)]
    dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)]

    files = cat_files + dog_files
    labels = [0] * len(cat_files) + [1] * len(dog_files)

    x_train, x_test, y_train, y_test = train_test_split(
        files, labels, test_size=0.2, shuffle=True
    )

    return x_train, x_test, y_train, y_test
