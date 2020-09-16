#-*-coding:utf-8-*- import torch
import torch
import torch.nn as nn

from torch.autograd import Variable

from train_config import config as cfg

class DistanceLoss(nn.Module):
    def __init__(self, ):
        super(DistanceLoss, self).__init__()



    def forward(self, x,y):


        dis_sum=torch.norm(x)*torch.norm(y)
        similiar=torch.sum(x*y)/dis_sum


        loss=1-similiar

        if loss<0:
            loss=0
        return loss



