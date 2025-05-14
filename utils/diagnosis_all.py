import torch
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc


# ---------------- Example usage ----------------
model_nih = xrv.models.DenseNet(weights="densenet121-res224-nih")
model = xrv.models.ResNet(weights="resnet50-res512-all")
print("nih\n",model.targets)
print("nih\n",model_nih.targets)


