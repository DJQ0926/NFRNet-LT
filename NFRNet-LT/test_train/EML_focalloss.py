import torch
import torch.nn as nn
import torch.nn.functional as F
from  config import *

class MultiLossFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', num_classes=37, beta=0.99):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.beta = beta
        self.class_freqs = torch.zeros(num_classes).to(DEVICE)
        self.class_weights = torch.zeros(num_classes).to(DEVICE)
        self.num_classes = num_classes

    def forward(self, input, target):
        
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        weights = torch.zeros_like(target, dtype=torch.float)
        for j in range(self.num_classes):
            mask = target == j
            freq = mask.sum().float()               
            self.class_freqs[j] += freq              
           
            weights[mask] = 1 / (self.beta ** freq)          
        self.class_weights = 1 / (self.beta ** (self.class_freqs / self.class_freqs.sum()))  

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        focal_loss = focal_loss * weights
        if self.reduction == 'mean':
            return (focal_loss.mean() / self.class_weights).mean()
        elif self.reduction == 'sum':
            return (focal_loss.sum() / self.class_weights).sum()
        else:
            return (focal_loss / self.class_weights).mean()