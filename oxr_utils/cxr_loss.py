import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchxrayvision as xrv
from torchmetrics import StructuralSimilarityIndexMeasure
from .cxr_processing import norm_to_xrv

class DiagnosticLoss(nn.Module):
    """
    Diagnostic loss for CXR images.
    This loss is used to train a model to predict the presence of certain diseases in chest X-ray images.
    The loss is based on the binary cross-entropy loss between the predicted and target labels.
    """
    
    def __init__(self, num_classes=14, device="cuda"):
        super(DiagnosticLoss, self).__init__()

        self.num_classes = num_classes
        self.device = device
        #self.diagnosis_model = xrv.models.ResNet(weights="resnet50-res512-all").to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()


    def forward(self, outputs, targets):
        """
        Compute the diagnostic loss.
        
        Args:
            outputs (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth labels.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure the outputs and targets are on the same device
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        if outputs.shape[-1] == 1024:
            outputs = F.avg_pool2d(outputs, 2)
        if targets.shape[-1] == 1024:
            targets = F.avg_pool2d(targets, 2)

        with torch.no_grad():
            # Rescale [0, 1] -> [-1024, 1024]
            outputs = norm_to_xrv(outputs)
            targets = norm_to_xrv(targets)

            outputs_dignosis = self.diagnosis_model(outputs)
            targets_diagnosis = self.diagnosis_model(targets)
            
        # Compute the binary cross-entropy loss
        diagnosis_loss = self.criterion(outputs_dignosis, targets_diagnosis)
        
        return diagnosis_loss

class SSIMLoss(nn.Module):
     
    def __init__(self, device="cuda"):
        super(SSIMLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, outputs, targets):

        # Ensure the outputs and targets are on the same device
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        # Compute the SSIM loss
        ssim_loss = 1 - self.ssim(outputs, targets)
        
        return ssim_loss

class GradientLoss(nn.Module):
    """
    Losstype = "mse" or "l1"
    """
    def __init__(self, device="cuda", loss_type="l1"):
        super(GradientLoss, self).__init__()
        self.device = device
        self.loss_type = ""
        
    def forward(self, outputs, targets):
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        pred_grad_x, pred_grad_y = torch.gradient(pred)
        target_grad_x, target_grad_y = torch.gradient(target)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        else:
            print("invalid losstype")
        return loss
    

class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, loss_weights=None, device="cuda"):
        super(CombinedLoss, self).__init__()
        if loss_weights is None:
            loss_weights = [1.0/len(loss_fns) for _ in range(len(loss_fns))]
        self.loss_weights = loss_weights
        self.loss_fns = [loss_fns.to(device) for loss_fn in loss_fns]
        

    def forward(self, outputs, targets):
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        total_loss = 0.0
        for loss_fn, loss_w in zip(self.loss_fns, self.loss_weights):
            total_loss += loss_w * loss_fn(outputs, targets)
        
        return total_loss

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss