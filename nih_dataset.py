# v1.0
# Based on torchxrayvision, used under the MIT-licence. 
import os
from skimage.io import imread
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


class NIH_Dataset(Dataset):
    """
    Loads data from NIH ChestX-ray14 dataset with updated CSV format:
    - 'Image Index' column contains full image paths (relative to dataset_root)
    - 'Train', 'Val', and 'Test' columns indicate split membership

    Parameters:
        dataset_root: Path to the dataset directory (not used for splits now)
        split: 'train', 'val', or 'test'
        img_root_override: Override for image folder (unused if paths are absolute)
        csvpath: Path to the CSV metadata file
        views: List of image views to include, or ["*"] for all
        unique_patients: If True, only one image per patient
        out_array_type: "torch" or "np"
        out_min, out_max: Normalized output image intensity range
        out_size: size of the image, if !=1024, image will be resized with interpolation
        img_max_val: Maximum intensity value in the raw image
        no_lbls: If True, only the image is returned
    """

    def __init__(self,
                 dataset_root,
                 split="train",
                 img_root_override=None,
                 csvpath=None,
                 views=["PA", "AP"],
                 unique_patients=False,
                 out_array_type="torch",
                 out_min=0.0,
                 out_max=1.0,
                 out_size=1024, # will bi-lerp to resize - only for torch
                 img_max_val=255.0,
                 no_lbls=False):

        super(NIH_Dataset, self).__init__()

        assert split in {"train", "val", "test"}, "Invalid split specified."
        self.split = split
        self.img_max_val = img_max_val
        self.out_min = out_min
        self.out_max = out_max
        self.out_size = out_size
        self.out_array_type = out_array_type
        self.no_lbls = no_lbls
        self.dataset_root = dataset_root

        self.pathologies = [
            "Atelectasis", "Consolidation", "Infiltration",
            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
            "Effusion", "Pneumonia", "Pleural_Thickening",
            "Cardiomegaly", "Nodule", "Mass", "Hernia"
        ]

        # Load CSV
        if csvpath is None:
            csvpath = os.path.join(dataset_root, "EnhancedDataEntry.csv")

        self.csv = pd.read_csv(csvpath)

        # Standardize columns
        self.csv["view"] = self.csv["View Position"].fillna("UNKNOWN")
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        self.csv["age_years"] = self.csv["Patient Age"] * 1.0
        self.csv["sex_male"] = self.csv["Patient Gender"] == 'M'
        self.csv["sex_female"] = self.csv["Patient Gender"] == 'F'

        # Filter by split
        split_col = {"train": "Train", "val": "Val", "test": "Test"}[split]
        self.csv = self.csv[self.csv[split_col] == "Yes"]

        # Filter by view
        self.views = views if isinstance(views, list) else [views]
        if "*" not in self.views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]

        # Unique patient filter
        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Store image paths
        self.img_paths = self.csv["Image Index"].apply(lambda p: p.replace("\\", "/")).tolist()

        # Compute labels
        self.labels = np.stack([
            self.csv["Finding Labels"].str.contains(p).fillna(False).values.astype(np.float32)
            for p in self.pathologies
        ], axis=1)

    def string(self):
        return f"{self.__class__.__name__} num_samples={len(self)} views={self.views} unique_patients={self.split}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.img_paths[idx])
        img = imread(img_path).astype(np.float32)

        if img.ndim > 2: # for the rare rgba images in the dataset
            img = np.mean(img, axis=2)

        # Normalize
        img = (img / self.img_max_val) * (self.out_max - self.out_min) + self.out_min
        img = img[None, :, :]  # add channel dim

        lbl = self.labels[idx]

        
        img = torch.from_numpy(img)
        lbl = torch.from_numpy(lbl)
        
        if self.out_size != 1024: # resize only if not the assumed size
            img = F.interpolate(img,(self.out_size, self.out_size),mode='bilinear')

        if self.out_array_type == "np":
            img = img.numpy()
            lbl = lbl.numpy()
        
        return img if self.no_lbls else (img, lbl)
    