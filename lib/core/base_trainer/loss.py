import torch
import math
import torch.nn as nn
import torch.nn.functional as F




class BCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth_eps=None):
        super(BCEWithLogitsLoss, self).__init__()






        self.smooth=smooth_eps

    def forward(self,input,target,training=True,features=None):
        if training:
            smooth_target = (
                    target * (1 - 2*self.smooth) +  self.smooth)

            loss=F.binary_cross_entropy_with_logits(input,smooth_target)
        else:
            if features is not None:



                control_index=(features[:,0]==0)

                loss = F.binary_cross_entropy_with_logits(input, target,reduction='none')

                loss=torch.mean(loss,dim=1)

                loss=loss*control_index.float()

                loss = torch.mean(loss)



        return loss



