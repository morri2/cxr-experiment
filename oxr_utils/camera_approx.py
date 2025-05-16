import torch
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
    
class CamearApproxInverse(nn.Module):
    """
    Undo the linear mapping of a CameraApprox
    x = (y - m) / k
    where y is input, m and k are from CameraApprox.
    """

    def __init__(self, camera_approx: CameraApprox = CameraApprox(1.0)):
        super().__init__()
        self.k = camera_approx.linear_map_k
        self.m = camera_approx.linear_map_m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x in [0, 1] or scaled accordingly
        out = (x - self.m) / self.k
        return out.clamp(0.0, 1.0)