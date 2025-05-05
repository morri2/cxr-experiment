# v1.0
# Based on torchxrayvision, used under the MIT-licence. 
import os
from skimage.io import imread
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class NIH_CxrLblDataset(Dataset):
    """
 Loads data from NIH ChestX-ray14 dataset.
    
Assumes dataset is stored in a structure like:
```
[dataset_root]         <- Provide path to this
| Data_Entry_2017.csv   
| test_list.txt
| train_val_list.txt
| images                (folder)
| | 00000001_000.png    
| | ...                 (all images)
```

## Parameters
    **dataset_root**: Path to the root folder of the dataset (see above).
    **img_root_override** = None: Optional path to override image folder location.
    **csvpath** = None: Optional path to override CSV metadata file.
    **views** = ["PA", "AP"]: List of image views to include (e.g. "PA", "AP"). "*" is wildcard.
    **test** = False: Use test set if True, otherwise train/val.
    **unique_patients** = False: If True, include only one image per patient.
    **out_array_type** = "torch": Output type: "torch" tensor or "np" array.
    **out_min** = 0.0: Minimum output image intensity after normalization.
    **out_max** = 1.0: Maximum output image intensity after normalization.
    **img_max_val** = 255.0: Max value in image file for normalization scaling.
    **no_lbls** = False: if true, only the *img* is returned insted of *cxr, lbls*

    """
    def __init__(self,
                 dataset_root,
                 img_root_override=None, # override
                 csvpath = None, # override
                 views=["PA", "AP"],
                 test=False, # True to get test dataset
                 unique_patients=False,
                 out_array_type="torch", # "torch" or "np"
                 out_min=0.0,
                 out_max=1.0, 
                 img_max_val=255.0, # highest value in stored images
                 no_lbls=False, # if true, only the cxr is returned insted of (cxr, lbls)
                 ):
        super(NIH_CxrLblDataset, self).__init__()

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                    "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                    "Effusion", "Pneumonia", "Pleural_Thickening",
                    "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        # imgpath
        if img_root_override is None:
            self.img_root = os.path.join(dataset_root, "images")
        else:
            self.img_root = img_root_override
        
        self.img_max_val = img_max_val 
        self.out_min=out_min
        self.out_max=out_max
        self.out_array_type=out_array_type
        self.no_lbls = no_lbls

        # Load csv
        if csvpath is None:
            csvpath = os.path.join(dataset_root, "Data_Entry_2017.csv")
        
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        
        # Rename and retype fields.
        # view
        self.csv['view'] = self.csv['View Position']

        # patientid
        self.csv['patientid'] = self.csv['Patient ID'].astype(str)

        # age
        self.csv['age_years'] = self.csv['Patient Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F' 


        # Dataset limits        
        self.test = test
        self.views = views if type(views) is list else [views]   # make sure self.views is list
        self.unique_patients = unique_patients

        self.csv["view"] = self.csv["view"].fillna("UNKNOWN")
        if "*" not in self.views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view


        # Train-Val/Test Filter
        split_filename = "test_list.txt" if self.test else "train_val_list.txt"
        split_path = os.path.join(dataset_root, split_filename)

        with open(split_path, "r") as f:
            image_list = set(line.strip() for line in f)

        self.csv = self.csv[self.csv["Image Index"].isin(image_list)] # Filter self.csv by image list


        # Unique patient filter
        if unique_patients:
            self.csv: pd.DataFrame = self.csv.groupby("Patient ID").first()
        self.csv = self.csv.reset_index() # ungroups


        # Get labels for imgs - must be done after filtering
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T # flip
        self.labels = self.labels.astype(np.float32)


    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} unique_patients={}, test={}".format(len(self), self.views, self.unique_patients, self.test)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.img_root, imgid)
        img = imread(img_path).astype(np.float32)

        # Format and normalize image to desired range
        if len(img.shape) > 2:
            img = np.mean(img, axis=2) # in case of rare rgba images

        img = (img / self.img_max_val) * (self.out_max - self.out_min) + self.out_min
        img = img[None, :, :]
        
        lbl = self.labels[idx]
        if self.out_array_type == "torch":
            img, lbl = torch.from_numpy(img), torch.from_numpy(lbl)

        if self.no_lbls:
            return img
        else:
            return img, lbl

class SubsetDataset(Dataset):
    """When you only want a subset of a dataset the `SubsetDataset` class can
    be used. A list of indexes can be passed in and only those indexes will
    be present in the new dataset. This class will correctly maintain the
    `.labels`, `.csv`, and `.pathologies` fields and offer pretty printing.

    .. code-block:: python

        dsubset = xrv.datasets.SubsetDataset(dataset, [0, 5, 60])
        # Output:
        SubsetDataset num_samples=3
        of PC_Dataset num_samples=94825 views=['PA', 'AP']

    For example this class can be used to create a dataset of only female
    patients by selecting that column of the csv file and using np.where to
    convert this boolean vector into a list of indexes.

    .. code-block:: python

        idxs = np.where(dataset.csv.PatientSex_DICOM=="F")[0]
        dsubset = xrv.datasets.SubsetDataset(dataset, idxs)
        # Output:
        SubsetDataset num_samples=48308
        - of PC_Dataset num_samples=94825 views=['PA', 'AP'] data_aug=None

    """

    def __init__(self, dataset, idxs):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = idxs

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]
    
def split_dataset_at(dataset, split_idx, cutoff=2_147_483_647):
    """Returns 2 datasets, split from the input at index, ignores data larger than cutoff (default is 32bit int max)"""
    if split_idx >= len(dataset):
        raise Exception("Split is outside of dataset") 
    a = SubsetDataset(dataset, range(split_idx))
    b = SubsetDataset(dataset, range(split_idx, min(len(dataset), cutoff)))
    return a, b