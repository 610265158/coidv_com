import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import numpy as np
import torch.nn as nn
from train_config import config as cfg

import torch
import torch.nn as nn
from sklearn.metrics import log_loss

from lib.helper.logger import logger
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LoglossMeter(object):
    def __init__(self,which_col=(0,206)):
        self.reset()
        self.col_range=which_col
    def reset(self):
        self.y_true = np.array([0 for i in range(206)])
        self.y_pred = np.array([0 for i in range(206)])
        self.score = 0

        self.item=1

        self.eps=1e-5
    def update(self,  y_pred,y_true):
        y_true = y_true
        y_pred = torch.sigmoid(y_pred)

        logloss=y_true*torch.log(y_pred+self.eps)+(1-y_true)*torch.log(1-y_pred)

        self.score+=torch.mean(logloss).cpu().item()


        self.item+=1

    @property
    def avg(self):

        return self.score/self.item

