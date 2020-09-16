
import numpy as np
import random
import torch

import torch.nn as nn
from train_config import config as cfg
from lib.core.model.loss.ohem import OHEMLoss
from lib.core.model.loss.labelsmooth import LabelSmoothing
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def cutmix_numpy(data, mask,targets, alpha):


    indices=list(range(data.shape[0]))
    random.shuffle(indices)
    # shuffled_data = data[indices]
    shuffled_targets = targets[indices]




    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    mask[:, :, bbx1//8:bbx2//8, bby1//8:bby2//8] = mask[indices, :, bbx1//8:bbx2//8, bby1//8:bby2//8]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.shape[-1] * data.shape[-2]))

    targets = [targets, shuffled_targets, lam]

    return data,mask, targets

def cutmix(data,targets, alpha):
    indices = torch.randperm(data.size(0))
    # shuffled_data = data[indices]
    shuffled_targets = targets[indices]


    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]


    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets, shuffled_targets, lam]
    return data, targets

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets, shuffled_targets, lam]

    return data, targets


def cutmix_criterion(preds1, targets,criterion):
    targets1, targets2,lam = targets[0], targets[1], targets[2]

    loss1 = lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)

    return loss1

def mixup_criterion(preds, targets,criterion):
    targets1, targets2, lam = targets[0], targets[1], targets[2]

    loss1=lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

    return loss1