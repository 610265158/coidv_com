import pandas as pd

import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt

from train_config import config as cfg
from lib.core.base_trainer.model import Complexer
from lib.dataset.dataietr import DataIter

data = pd.read_json('folds.json',lines=True)

cfg.TRAIN.batch_size=1
pre_length=91
fold=0
###build dataset
train_data=data[data['fold']!=fold]
val_data=data[data['fold']==fold]

train_ds=DataIter(train_data,shuffle=True,training_flag=True)
val_ds=DataIter(val_data,shuffle=True,training_flag=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

short_model=Complexer(pre_length=pre_length)


weight='./fold0_epoch_138_val_loss0.250854.pth'


short_model.load_state_dict(torch.load(weight, map_location=device))
short_model.to(device)


for i in range(val_data.size):
    images_np, data, target = val_ds()
    images = torch.from_numpy(images_np).to(device).float()
    data = torch.from_numpy(data).to(device).float()

    pre=short_model(images,data)[0]

    pre=pre.data.cpu().numpy()

    target=target[0][:pre_length,...]

    print(pre.shape)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(range(pre_length), pre[...,0], color='blue', label='pre', linewidth=0.8) #
    plt.plot(range(pre_length), target[...,0], color='red', label='target', linewidth=0.8)  #

    ax2 = plt.subplot(3, 1, 2)
    plt.plot(range(pre_length), pre[...,1], color='blue', label='pre', linewidth=0.8)  #
    plt.plot(range(pre_length), target[...,1], color='red', label='target', linewidth=0.8)  #

    ax3 = plt.subplot(3, 1, 3)
    plt.plot(range(pre_length), pre[...,3], color='blue', label='pre', linewidth=0.8)  #
    plt.plot(range(pre_length), target[...,3], color='red', label='target', linewidth=0.8)  #

    plt.legend()
    plt.show()

    cv2.namedWindow('ss',0)

    cv2.imshow('ss',images_np[0,0,...])
    cv2.waitKey(0)

    plt.close('all')
##predict


