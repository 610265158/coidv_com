import torch
import math
import torch.nn as nn
import torch.nn.functional as F




class BCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth_eps=None):
        super(BCEWithLogitsLoss, self).__init__()






        self.smooth=smooth_eps

    def forward(self,input,target):
        smooth_target = (
                target * (1 - 2*self.smooth) +  self.smooth)

        loss=F.binary_cross_entropy_with_logits(input,smooth_target)

        return loss


