import warnings

import sklearn

warnings.filterwarnings('ignore')

#the basics
import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm



#for model evaluation
from sklearn.model_selection import train_test_split, KFold




SNR_THRS=1.2

ROOT='../stanford-covid-vaccine'
train = pd.read_json(ROOT+'/train.json', lines=True)

train['fold']=-1

kf=sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

for fold,(train_index , test_index) in enumerate(kf.split(train)):
    train['fold'][test_index]=fold


print(train['fold'])

save = pd.DataFrame(train)
save.to_json('folds.json',lines=True,orient='records')
