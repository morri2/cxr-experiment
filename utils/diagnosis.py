import torch
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc

from .cxr_dataset import *
from .cxr_processing import *
from .auc_roc_utils import *   

class NihTester:

    def __init__(self, cxr_label_dataset: Dataset, device="cuda"):
        self.diagnosis_model = xrv.models.DenseNet(weights="densenet121-res224-nih").to(device)
        self.device = device
        self.dataloader = DataLoader(cxr_label_dataset, batch_size=16, shuffle=False)
        self.criterion = torch.nn.L1Loss().to(device)

        self.cache_all_outputs = None
        self.cache_all_labels = None


    def gather_outputs_labels_module(self, module):
        """Predict all outputs and gather ground truth labels. Feeds inputs thru module first (can be used to test noise or noise-denoise chains)""" 

        self.diagnosis_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloader, desc="Gathering outputs/labels"):

                inputs = module(inputs.to(self.device)) # execute module on inputs

                inputs = norm_to_xrv(inputs)
                inputs = scale_and_crop(inputs, 224)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)[:, :14]  # NIH has 14 labels
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        self.cache_all_outputs = torch.cat(all_outputs, dim=0).numpy()
        self.cache_all_labels = torch.cat(all_labels, dim=0).numpy()

        return self.cache_all_outputs, self.cache_all_labels

    def roc_module(self, module):
        o, l = self.gather_outputs_labels_module(module)

    def _gather_outputs_labels(self, use_cache=False):
        """Helper function: predict all outputs and gather ground truth labels.""" 
        if use_cache and self.cache_all_outputs is not None and self.cache_all_labels is not None:
            return self.cache_all_outputs, self.cache_all_labels

        self.diagnosis_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloader, desc="Gathering outputs/labels"):
                inputs = norm_to_xrv(inputs)
                inputs = scale_and_crop(inputs, 224)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)[:, :14]  # NIH has 14 labels
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        self.cache_all_outputs = torch.cat(all_outputs, dim=0).numpy()
        self.cache_all_labels = torch.cat(all_labels, dim=0).numpy()

        return self.cache_all_outputs, self.cache_all_labels


    def diagnosis_auc(self, use_cache=True):
        all_outputs, all_labels = self._gather_outputs_labels(use_cache=use_cache)

        aucs = []
        for i in range(all_labels.shape[1]):
            try:
                auc_val = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            except ValueError:
                auc_val = np.nan  # Handle case with no positives/negatives
            aucs.append(auc_val)

        return np.array(aucs)


    def avg_diagnosis_auc(self, use_cache=True):
        aucs = self.diagnosis_auc(use_cache=use_cache)
        if np.isnan(aucs).any():
            print("\nWarning: Some AUCs are NaN (likely from label imbalance).")
        return np.nanmean(aucs)

    def plot_roc_curve(self, label_idx, use_cache=True):
        """Plot ROC curve for a single diagnosis."""
        all_outputs, all_labels = self._gather_outputs_labels(use_cache=use_cache)
        label_names = self.diagnosis_model.pathologies[:14]

        if np.unique(all_labels[:, label_idx]).size < 2:
            print(f"Cannot plot ROC for {label_names[label_idx]}: not enough positives/negatives.")
            return

        fpr, tpr, _ = roc_curve(all_labels[:, label_idx], all_outputs[:, label_idx])
        fig, ax = setup_roc_plot(title=f"ROC Curve: {label_names[label_idx]}")
        add_roc_curve(ax, fpr, tpr, label=label_names[label_idx], color='blue', auc_in_label=True)
        finalize_roc_plot(ax)




    def plot_all_roc_curves_single_plot(self, use_cache=True):
        """Plot all ROC curves together in a single plot."""
        all_outputs, all_labels = self._gather_outputs_labels(use_cache=use_cache)
        label_names = self.diagnosis_model.pathologies[:14]

        fig, ax = setup_roc_plot("All ROC Curves: ChestX-ray14 Diagnoses")

        for i in range(all_labels.shape[1]):
            if np.unique(all_labels[:, i]).size < 2:
                continue

            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            add_roc_curve(ax, fpr, tpr, label=label_names[i], alpha=0.85, auc_in_label=True)

        finalize_roc_plot(ax, loc="lower right", ncol=2)


    def plot_avg_roc_curve(self, use_cache=True, num_points=100):
        """Plot the average ROC curve across all diagnoses (no std shading)."""
        all_outputs, all_labels = self._gather_outputs_labels(use_cache=use_cache)
        label_names = self.diagnosis_model.pathologies[:14]

        mean_fpr = np.linspace(0, 1, num_points)
        tprs = []
        aucs = []

        for i in range(all_labels.shape[1]):
            if np.unique(all_labels[:, i]).size < 2:
                continue

            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            aucs.append(auc(fpr, tpr))
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        if not tprs:
            print("No valid ROC curves to average.")
            return

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        fig, ax = setup_roc_plot("Average ROC Curve Across Diagnoses")
        add_roc_curve(ax, mean_fpr, mean_tpr, label="Mean ROC", color='blue', auc_in_label=True)
        finalize_roc_plot(ax)


# ---------------- Example usage ----------------


# nih = xrv.datasets.NIH_Dataset("../../data/NIH")
# nih_tester = NihTester(CxrLabelDataset(xrv.datasets.SubsetDataset(nih, range(4000))), device="cuda")

# print(nih_tester.avg_diagnosis_auc())

# nih_tester.plot_avg_roc_curve()
# plt.show()

# nih_tester.plot_all_roc_curves_single_plot()   # Full nice plot
# plt.show()