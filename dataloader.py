from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple, Optional, Callable

IMG_DIR = "directory path"
MASK_DIR = "other directory path"

class PanoramaDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, transform: Optional[Callable]):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform




    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        name = "1{idx:04d}_00001.nii.gz"
        img = f'{self.img_dir}/{name}'
        mask = f'{self.mask_dir}/{name}'
