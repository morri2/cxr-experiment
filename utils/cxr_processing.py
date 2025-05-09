import torchxrayvision as xrv
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

# ------ CXR Value Ranges ------
def to_range(cxr, from_min, from_max, to_min, to_max):
    """Scales images in [from_min, from_max] to be in the range [to_min, to_max]."""
    cxr = ((cxr - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
    cxr = cxr.clamp(to_min, to_max)
    return cxr

def norm_to_xrv(cxr: Tensor):
    """Scales images to be [-1024 1024]. the range used in xrv."""
    return to_range(cxr, 0.0, 1.0, -1024, 1024)

def xrv_to_norm(cxr: Tensor):
    """Scales images to be [-1024 1024]. the range used in xrv."""
    return to_range(cxr, -1024, 1024, 0.0, 1.0)

def normalize(cxr: Tensor):
    """normalizes the image to the range [0,1]"""
    minval = cxr.min()
    maxval = cxr.max()
    return to_range(cxr, minval, maxval, 0.0, 1.0)

# ------ CXR Domain ------

def to_log_cxr(T, eps=1e-6):
    return -torch.log(T.clamp(min=eps))  # clamp to avoid log(0)

def to_transmission(L):
    return torch.exp(-L)

# ------ CXR Image Dimensions ------


INTERPOLATION_DICT = {
    "nearest_exact": transforms.InterpolationMode.NEAREST_EXACT,
    "nearest": transforms.InterpolationMode.NEAREST,
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
}

def scale_and_crop(img: Tensor, side=224, interpolation="nearest_exact"):
    """Scale and crop the input image to side x side. default: side=224"""
    img = transforms.Resize(side, interpolation=INTERPOLATION_DICT[interpolation])(img) # smallest edge will be side
    img = transforms.CenterCrop(side)(img) # crop to side x side
    return img

def fix_dims(img: Tensor):
    """Fix the dimensions to 4D"""
    x, y = img.shape[-2], img.shape[-1]
    return img.reshape((1,1,x,y))

# ------ Noise ------
def noise(cxr: Tensor, strength: float = 0.3, min_signal: float = 0.01) -> Tensor:
    """Noise to [0,1] image. std_dev = strength * signal. min_signal serves to avoid 0 noise."""
    cxr = cxr.clamp(min_signal)
    return torch.normal(cxr, std= strength * cxr).clamp(0, 1)

class NoiseModule(nn.Module):
    """Adds noise to the input image."""
    def __init__(self, strength: float = 0.3, min_signal: float = 0.01) -> None:
        super().__init__()
        self.strength = strength
        self.min_signal = min_signal

    def forward(self, cxr: Tensor) -> Tensor:
        return noise(cxr, self.strength, self.min_signal)


# ------ CameraDistort ------
from torchvision.transforms import GaussianBlur


class CameraDistort(nn.Module):
    def __init__(self, noise_std_const=0.2, noise_std_factor=0.1, blur_sigma=3.0):
        # NOISE PARAMETERS
        super().__init__()
        self.noise_std_const = noise_std_const
        self.noise_std_factor = noise_std_factor
        self.blur_sigma = blur_sigma


        self.blur_kernel_size = (int) (self.blur_sigma * 8)
        if self.blur_kernel_size % 2 == 0: # make sure its odd
            self.blur_kernel_size += 1

        self.blur = GaussianBlur(self.blur_kernel_size, self.blur_sigma)


    def forward(self, x):
        # blur
        x = self.blur(x)
        # noise
        x = torch.normal(x, std= (x + self.noise_std_const) * self.noise_std_factor)
        return x.clamp(0.0, 1.0)

# ------ Denoise / Filtering ------

class MedianFilter(nn.Module):
    def __init__(self, device, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device

    def forward(self, x):
        x = median_filter_lambda(x.to(self.device), self.kernel_size)
        return x.to(self.device)

median_filter_lambda = lambda x, k: F.pad(x.unfold(-2, k, 1).unfold(-2, k, 1).flatten(-2).median(-1)[0], (k // 2,) * 4, "replicate")

class MeanFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        x = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return x