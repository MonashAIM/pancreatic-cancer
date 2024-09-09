from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

IMG_DIR = "./data"
LABEL_DIR = "./data/clinical_information.csv"
MASK_DIR = "./data/mask"

class PanoramaDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, label_dir: str, transform: Optional[Callable] = None):
        """Initialise dataset with image directories and labels"""
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.labels = pd.read_csv(label_dir)
        self.transform = transform

    def load_image(self, name: str, dir: str) -> Tensor:
        """Helper to load nibabel image, from dir/name"""
        # Load image
        name = f'{dir}/{name}'
        img = nib.load(name).get_fdata().astype(np.float32) # type: ignore

        # Format to (channel, depth, width, height)
        image = torch.from_numpy(img)
        image = torch.permute(image, (2, 0, 1))
        image = image.unsqueeze(0)
        return image


    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        """
        Get original image, segmentation mask and lesion label for
        a given index.
        """
        # Get Segmentation Image
        index = f"1{idx:05d}_00001"
        mask_name = f"{index}.nii.gz"
        mask_image = self.load_image(mask_name, self.mask_dir)

        # Get Image
        name = f"{index}_0000.nii.gz"
        image = self.load_image(name, self.img_dir)
        if self.transform:
            image = self.transform(image)

        # Get Label
        label = self.labels.iloc[idx, -2]
        return image, mask_image, label

def get_image(tensor: Tensor) -> np.ndarray:
    """Helper to convert tensor to numpy array"""
    return tensor.numpy().squeeze()

if __name__ == "__main__":
    # Add data preprocessing to transform
    dataset = PanoramaDataset(IMG_DIR, MASK_DIR, LABEL_DIR, transform=None)

    # Example of indexing into the dataset
    img, mask, label = dataset[546]

    # If you want to display an image using matplotlib
    plt.imshow(get_image(img)[32])
    plt.show()
