import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np
import pdb
import torch.nn.functional as F

class UltrasoundImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(csv_file, header=[0])[["id", "label"]]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        # pdb.set_trace()

        return image, label
