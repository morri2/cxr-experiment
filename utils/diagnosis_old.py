import torch
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


from cxr_dataset import *
from cxr_processing import *

IDENTITY = torch.nn.Identity()


class NihTester:
    def __init__(self, cxr_label_dataset: CxrLabelDataset, device="cuda"):
        # Note, 14 first targets are valid!
        self.diagnosis_model = xrv.models.DenseNet(weights="densenet121-res224-nih").to(device)
        self.device = device
        self.dataloader = DataLoader(cxr_label_dataset, batch_size=16, shuffle=False)
        self.criterion = torch.nn.L1Loss().to(device)
        
    
    def diagnosis_auc(self):
        self.diagnosis_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                inputs, labels = batch

                inputs = norm_to_xrv(inputs)
                inputs = scale_and_crop(inputs, 224)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)
                outputs = outputs[:, :14]  # 14 for NIH

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        print(np.sum(all_labels, axis=0))  # number of positives per disease
        print(np.sum(1 - all_labels, axis=0))  # number of negatives per disease

        aucs = []
        for i in range(all_labels.shape[1]):
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            aucs.append(auc)

        return np.array(aucs)
    
    def avg_diagnosis_auc(self):
        diagnosis_auc = self.diagnosis_auc()
        if np.isnan(diagnosis_auc).any():
            print("\n!AUCs has one or more NaN values. Likely due to all 1s/0s in dataset. Use a larger dataset!")
        return np.nanmean(self.diagnosis_auc())


    # ------ ROC AUC Curve Plots ------
        
    def plot_roc_curve(self):
        self.diagnosis_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                inputs, labels = batch

                inputs = norm_to_xrv(inputs)
                inputs = scale_and_crop(inputs, 224)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)
                outputs = outputs[:, :14]  # 14 for NIH

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Pick a single disease label to plot (for example, the first disease: index 0)
        label_idx = 0

        # Check if label_idx has enough positive/negative samples
        if np.unique(all_labels[:, label_idx]).size < 2:
            print(f"Cannot plot ROC for label {label_idx}: not enough positive/negative samples.")
            return

        fpr, tpr, thresholds = roc_curve(all_labels[:, label_idx], all_outputs[:, label_idx])
        roc_auc = auc(fpr, tpr)

        # Plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve for label {label_idx}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_all_roc_curves(self):
        self.diagnosis_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                inputs, labels = batch

                inputs = norm_to_xrv(inputs)
                inputs = scale_and_crop(inputs, 224)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)
                outputs = outputs[:, :14]  # NIH has 14 labels

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        label_names = self.diagnosis_model.pathologies[:14]

        num_labels = all_labels.shape[1]
        cols = 4
        rows = (num_labels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()

        for i in range(num_labels):
            ax = axes[i]

            if np.unique(all_labels[:, i]).size < 2:
                ax.set_axis_off()
                ax.text(0.5, 0.5,
                        f"{label_names[i]}\n(Not enough data)",
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=10,
                        transform=ax.transAxes)
                continue

            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('1-Specificity')
            ax.set_ylabel('Sensitivity')
            ax.set_title(f"{label_names[i]}")
            ax.legend(loc="lower right")
            ax.grid(True)

        # Hide any extra unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()

        
        plt.tight_layout(pad=4.0)
        plt.show()

    def plot_all_roc_curves_single_plot(self):
        self.diagnosis_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                inputs, labels = batch

                inputs = norm_to_xrv(inputs)
                inputs = scale_and_crop(inputs, 224)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)
                outputs = outputs[:, :14]

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        label_names = self.diagnosis_model.pathologies[:14]

        plt.figure(figsize=(10, 10))

        for i in range(all_labels.shape[1]):
            if np.unique(all_labels[:, i]).size < 2:
                continue  # Skip if only positives or only negatives

            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'{label_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')  # Diagonal random line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curves for ChestX-ray14 Diagnoses')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True)
        plt.show()

nih = xrv.datasets.NIH_Dataset("../../data/NIH")

nih_tester = NihTester(CxrLabelDataset(xrv.datasets.SubsetDataset( nih, range(2000))), device="cuda")
print(nih_tester.avg_diagnosis_auc())
