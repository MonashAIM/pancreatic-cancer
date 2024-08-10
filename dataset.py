import os
from torch.utils.data import Dataset
import nibabel as nib
from torch import Tensor
from typing import Callable, Optional

class Isles22Dataset(Dataset):
    """
    Isles 2022 Dataset, written as a dataset subclass. Uses Nibabel to load 
    the images.
    
    Args:
        data_type (str): Type of data to load. One of (dwi, adc, flair).
        data_dir (str): Path to the data directory.
        transform (callable, optional): Optional transform to be applied

    Returns:
        MRI image of the respective type in the form of a Tensor
    """

    def __init__(self, data_type:str, data_dir:str='data', transform:Optional[Callable]=None):
        assert data_type in ('dwi', 'adc', 'flair'), 'Invalid data type not in (dwi, adc, flair)'
        self.data_dir = data_dir
        self.transform = transform
        self._num_samples = len(os.listdir(os.path.join(data_dir, 'rawdata')))
        self.data_type = data_type

    def __len__(self) -> int:
        return self._num_samples
    
    def get_data_path(self, index:int):
        assert index >= 0 and index < len(self), 'Index out of range'
        return os.path.join(self.data_dir, 'rawdata', f'sub-strokecase{index+1:04d}','ses-0001',f'sub-strokecase{index+1:04d}_ses-0001_{self.data_type}.nii.gz')
    
    def get_header(self, index:int):
        data_path = self.get_data_path(index)
        return nib.load(data_path).header

    def __getitem__(self, index:int) -> tuple[Tensor, Tensor]:
        data_path = self.get_data_path(index)
        mask_path = data_path.replace('rawdata','derivatives').replace(self.data_type,'msk')
        image = nib.load(data_path).get_fdata().astype('float32')
        mask = nib.load(mask_path).get_fdata().astype('float32')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
"""
Required Modifications to the data:
Sizes of the images are not the same.
All images need to be converted to float64 for dwi & flair and float32 for adc.
"""