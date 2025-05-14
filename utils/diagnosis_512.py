import torch
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torchvision.transforms as transforms

# ------ CXR Value Ranges ------
def to_range(cxr, from_min, from_max, to_min, to_max):
    """Scales images in [from_min, from_max] to be in the range [to_min, to_max]."""
    cxr = ((cxr - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
    cxr = cxr.clamp(to_min, to_max)
    return cxr

def norm_to_xrv(cxr: torch.Tensor):
    """Scales images to be [-1024 1024]. the range used in xrv."""
    return to_range(cxr, 0.0, 1.0, -1024, 1024)

def xrv_to_norm(cxr: torch.Tensor):
    """Scales images to be [-1024 1024]. the range used in xrv."""
    return to_range(cxr, -1024, 1024, 0.0, 1.0)

def normalize(cxr: torch.Tensor):
    """normalizes the image to the range [0,1]"""
    minval = cxr.min()
    maxval = cxr.max()
    return to_range(cxr, minval, maxval, 0.0, 1.0)



class ROCs:
    def __init__(self, preds, labels, targets=None):
        self.preds = preds
        self.labels = labels

        self.targets = targets

    def roc(self, idx):
        return roc_curve(self.labels[:,idx], self.preds[:,idx])
    
    def roc_fpr_tpr(self, idx):
        fpr, tpr, _ = roc_curve(self.labels[:,idx], self.preds[:,idx])
        return (fpr, tpr)
    
    def auc_score(self, idx):
        return roc_auc_score(self.labels[:,idx], self.preds[:,idx])   

    def total_auc_score(self):
        return roc_auc_score(self.labels, self.preds, average='micro')
    

class NihTester:
    """
    512x512 Tester, for nih dataset (only looks at 14 pathologies). 
    downscale: "2x2mean" / "scale_nearest" / "scale_bilinear" / "keep"
    
    """
    def __init__(self, cxr_label_dataset: Dataset, device="cuda", downscale="2x2mean"):
        self.device = device
        self.diagnosis_model = xrv.models.ResNet(weights="resnet50-res512-all").to(self.device)
      
        self.dataloader = DataLoader(cxr_label_dataset, batch_size=16, shuffle=False)

        self.downscale = nn.Identity()

        if downscale == "2x2mean":
            self.downscale = nn.AvgPool2d(2).to(self.device)
        elif downscale == "keep":
            self.downscale = nn.Identity().to(self.device)
        elif downscale == "scale_nearest":
            self.downscale = transforms.Resize(512, transforms.InterpolationMode.NEAREST).to(self.device)
        elif downscale == "scale_bilinear":
            self.downscale = transforms.Resize(512, transforms.InterpolationMode.BILINEAR).to(self.device)
        else:
            print("Invalid downscale settings!")



    def rocs(self, preproc: nn.Module = nn.Identity()):
        
        all_outputs = []
        all_labels = []


        try:
            preproc_device = next(preproc.parameters()).device
        except:
            preproc_device = "cpu"

        self.diagnosis_model.eval()
        preproc.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloader, desc="Making ROCs"):

                inputs = preproc(inputs.to(preproc_device))
                
                
                inputs = self.downscale(inputs.to(self.device))

                inputs = norm_to_xrv(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)[:, :14]  # NIH has 14 labels
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs, all_labels = torch.cat(all_outputs, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()

        return ROCs(all_outputs, all_labels, targets=self.diagnosis_model.targets)



if __name__ == "__main__":
    from nih_dataset import NIH_Dataset
    import camera_approx
    import auc_roc_utils
    import matplotlib.pylab as plt


    print("test!")
    dataset = NIH_Dataset("../data/NIH_data", split="test")
    print(len(dataset))
    tester = NihTester(dataset)
    print(tester.diagnosis_model.targets)

    rocs_noisy = tester.rocs(camera_approx.CameraApprox(1.0))
    print("noisy:", rocs_noisy.total_auc_score())
    rocs_clean = tester.rocs()
    print("clean:", rocs_clean.total_auc_score())
    
    fig, ax = auc_roc_utils.setup_roc_plot("Noisy vs Clean ROC")
    


    for i in range(14):
        print(f"{rocs_noisy.targets[i] if rocs_noisy.targets else '<none>'}: AUC clean: {rocs_clean.auc_score(i)}, AUC noisy: {rocs_noisy.auc_score(i)}")
        auc_roc_utils.add_roc_curve(ax, rocs_clean.roc_fpr_tpr(i), f"Clean {rocs_noisy.targets[i] if rocs_noisy.targets else ''}", color=auc_roc_utils.COLORS[i],  linestyle=":")
        auc_roc_utils.add_roc_curve(ax, rocs_noisy.roc_fpr_tpr(i), f"Noisy {rocs_noisy.targets[i] if rocs_noisy.targets else ''}", color=auc_roc_utils.COLORS[i], linestyle="-")

    auc_roc_utils.finalize_roc_plot(ax)
    plt.show()



    

    