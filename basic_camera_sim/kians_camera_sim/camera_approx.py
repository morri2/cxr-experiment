import torch
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LinearRegression
from torchvision import transforms
import torch.functional as F
import torch.nn as nn
import torch


class CameraApprox(nn.Module):
    """
    Model for approximating what a CXR (from a dataset like NIH) would look like taken with a camera-based sensor.
    Note: The model is limited in its ability to recreate information lost processing of the input CXR.
    This is most evident when it comes to empty space which by this model will have a transmission much lower than the expected ~100%.

    Input images are expected to be [0, arg:input_max]
    Outputs are [0, 1]
    Image noise and blur is calibrated for images ~ 1024x1024 in size
    """

    def __init__(self, input_max=255.0, output_max=None):
        super(CameraApprox, self).__init__()
        
        self.input_max = input_max
        self.output_max = output_max if output_max else input_max
        self.blur_sigma = 1.5
        self.linear_map_k = 0.5622 # same no matter domain
        self.linear_map_m = 0.43 # in assumed [0,1]
        self.noise_std = 0.031

        self.blur = transforms.GaussianBlur(11, self.blur_sigma)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x /= self.input_max
        x = self.blur(x)
        x = x * self.linear_map_k + self.linear_map_m
        x = torch.normal(x, self.noise_std)
        x *= self.output_max
        x = x.clip(0, self.output_max)
        return x
        
        


# Auto-Diagnosis as evaluation of ML-restoration of CXR images

# === In [0, 255] -> [0, 255] ===
# 
# Gaussian Blur: sigma=1.5 (kernel_size=11)
# Linear mapping: y = 0.5622 * x + 109.8140
# Gaussian Noise: std=8
#
# === In [0,1] -> [0,1] ===
# 
# Gaussian Blur: sigma=1.5 (kernel_size=11)
# Linear mapping: y = 0.5622 * x + 0.43
# Gaussian Noise: std = 0.031
#

if __name__ == "__main__":
    nih = Image.open('nih.png')
    nih = torch.from_numpy(np.array(nih)).unsqueeze(0)
    print(nih)

    mod = CameraApprox()
    out = mod(nih)

    vmin = 0
    vmax = 255

    print(nih.max())

    plt.subplot(1, 2, 1)
    plt.imshow(nih.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('in')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(out.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('out')
    plt.axis('off')

    plt.tight_layout()
    plt.show()






