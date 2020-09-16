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



class COVIDGenerator:
    def __init__(self,x,label,batch_size=32,training=False):
        self.data=x
        self.label=label
        self.batch_size=batch_size
        self.training=training


        self.size= self.__len__()

    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __call__(self):

        while 1:
            for i in range(self.size):
                if i==self.size-1:
                    cur_batch=self.data[i*self.batch_size:]
                    cur_label=self.label[i*self.batch_size:]
                else:
                    cur_batch = self.data[i * self.batch_size:(i+1) * self.batch_size]
                    cur_label = self.label[i * self.batch_size:(i+1) * self.batch_size]


                yield cur_batch,cur_label



class COVIDDataiter():
    def __init__(self,x,label,batch_size=16,training=False):

        self.generator=COVIDGenerator(x,label,batch_size,training)
        self.size=self.generator.size

    def __call__(self):
        xx=next(self.generator())



        return xx





train_ds=COVIDDataiter(train_inputs,train_labels,32,True)
val_ds=COVIDDataiter(val_inputs,val_labels,64,True)