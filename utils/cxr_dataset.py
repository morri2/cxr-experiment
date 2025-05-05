import itertools
import torchxrayvision as xrv
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import torchinfo
from .cxr_processing import *
    

class CxrDataset(Dataset):
    """returns cxr from dataset. Fixed up (cropped, scaled, etc.)"""
    
    def __init__(self, xrv_dataset, img_shape=224) -> None:
        super().__init__()
        self.xrv_dataset = xrv_dataset
        self.img_shape = img_shape

    def __len__(self):
        return len(self.xrv_dataset)
    
    def __getitem__(self, index) -> Tensor:
        cxr = scale_and_crop(torch.from_numpy(self.xrv_dataset[index]["img"]), side=self.img_shape)
        cxr = xrv_to_norm(cxr)
        return cxr

class CxrLabelDataset(Dataset):
    """returns (cxr, diagnosis_labels) from dataset. Fixed up (cropped, scaled, etc.)"""
    
    def __init__(self, xrv_dataset, img_shape=224) -> None:
        super().__init__()
        self.xrv_dataset = xrv_dataset
        self.img_shape = img_shape

    def __len__(self):
        return len(self.xrv_dataset)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        cxr = scale_and_crop(torch.from_numpy(self.xrv_dataset[index]["img"]), side=self.img_shape)
        cxr = xrv_to_norm(cxr)
        labels = torch.from_numpy(self.xrv_dataset[index]["lab"])
        return cxr, labels