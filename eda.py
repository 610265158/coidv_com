import warnings
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
test = pd.read_json(ROOT+'/test.json', lines=True)
sample_sub = pd.read_csv(ROOT+'/sample_submission.csv')


#target columns
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    encode=np.array(df[cols]
                            .applymap(lambda seq: [token2int[x] for x in seq])
                            .values
                            .tolist()
        )


    return encode

train_inputs = preprocess_inputs(train[train.signal_to_noise > SNR_THRS])
train_labels = np.array(train[train.signal_to_noise > SNR_THRS][target_cols].values.tolist())



print(train_inputs.shape)
print(train_labels.shape)

train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels,
                                                                     test_size=.1, random_state=34)





