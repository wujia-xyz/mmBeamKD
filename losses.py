"""
Loss functions for TransFuser-v3 beam prediction.
- FocalLoss: sigmoid focal loss for soft beam labels (only loss needed)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        if len(target.shape) == 1:
            target = F.one_hot(target, num_classes=64)
        return torchvision.ops.sigmoid_focal_loss(
            input, target.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
