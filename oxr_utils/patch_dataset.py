import torch
from torch.utils.data import Dataset, DataLoader
import random

class RandomPatchDataset(Dataset):
    def __init__(self, dataset, patch_size):
        """
        Args:
            dataset: the original dataset, patches of image (but not labels) will be returned
            patch_size: int or tuple
        """
        self.dataset = dataset  # This can be any dataset that returns torch tensors
        
        if type(patch_size) is not tuple:
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image and label from the pre-existing dataset (assuming tensor format)
        image, label = self.dataset[idx]

        # Get image dimensions: [C, H, W] format for tensors
        _, height, width = image.shape
        patch_height, patch_width = self.patch_size

        # Randomly select the top-left corner of the patch
        top = random.randint(0, height - patch_height)
        left = random.randint(0, width - patch_width)

        # Extract the patch by slicing the tensor
        patch = image[:, top:top + patch_height, left:left + patch_width]

        return patch, label


