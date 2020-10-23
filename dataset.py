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



feature_file='../lish-moa/train_features.csv'
target_file='../lish-moa/train_targets_scored.csv'
noscore_target='../lish-moa/train_targets_nonscored.csv'



train_features=pd.read_csv(feature_file)
labels_train=pd.read_csv(target_file)
extra_labels_train=pd.read_csv(noscore_target)

###ctrl_vehicle have no moa , set 0 directly in submission


###filter by ctl_vehicle
non_ctl_idx = train_features.loc[train_features['cp_type']!='ctl_vehicle'].index.to_list()
train_features = train_features.iloc[non_ctl_idx]
labels_train = labels_train.iloc[non_ctl_idx]
extra_labels_train = extra_labels_train.iloc[non_ctl_idx]



##encode cp_dose
dose=np.array(train_features['cp_dose'].values=='D1',dtype=np.float32)

train_features['cp_dose_encoded']=dose


train_features = train_features.drop(['sig_id','cp_type','cp_dose'],axis=1)

labels_train = labels_train.drop('sig_id',axis=1)
extra_labels_train=extra_labels_train.drop('sig_id',axis=1)


print(train_features.shape)
print(labels_train.shape)
print(extra_labels_train.shape)



print(labels_train.columns.values)

label_cnt_dict={}


for item in labels_train.columns.values:
    label_cnt_dict[item]=np.sum(labels_train[item])


label_cnt_dict_sorted=sorted(label_cnt_dict.items(), key=lambda x: x[1],reverse=True)


for k ,v in label_cnt_dict_sorted:
    print('%-30s: %d sample'%(k,v))





