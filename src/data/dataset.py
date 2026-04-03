"""
GTSRB PyTorch Dataset
=====================
Custom Dataset class for the German Traffic Sign Recognition Benchmark.
Handles image loading, CLAHE contrast enhancement, and augmentation.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """PyTorch Dataset for GTSRB traffic sign images.
    
    Args:
        dataframe: pandas DataFrame with 'Path' and 'ClassId' columns
        root_dir: path to the dataset root (containing Train/ and Test/)
        transform: torchvision transforms to apply
        apply_clahe: whether to apply CLAHE contrast enhancement
    """

    def __init__(self, dataframe, root_dir, transform=None, apply_clahe=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Path'])

        # Read image with OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE for contrast enhancement
        if self.apply_clahe:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['ClassId'], dtype=torch.long)
        return image, label
