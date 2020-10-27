import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.cluster import KMeans
from torch import nn

from lib.core.base_trainer.net_work import Train

from lib.core.model.ShuffleNet_Series.ShuffleNetV2.network import ShuffleNetV2

from lib.core.model.semodel.SeResnet import se_resnet50
import cv2
import numpy as np
import pandas as pd, numpy as np
from train_config import config as cfg
import setproctitle

from lib.dataset.dataietr import DataIter
setproctitle.setproctitle("alaska")

from train_config import seed_everything
from lib.helper.logger import logger

import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from lib.core.base_trainer.model import Complexer
from lib.core.base_trainer.densenet import Denseplexer
def main():


    train_feature_file = '../lish-moa/train_features.csv'
    target_file = '../lish-moa/train_targets_scored.csv'
    noscore_target = '../lish-moa/train_targets_nonscored.csv'

    train_features = pd.read_csv(train_feature_file)
    labels = pd.read_csv(target_file)
    extra_labels = pd.read_csv(noscore_target)

    test_features = pd.read_csv('../lish-moa/test_features.csv')



    ####





    print(train_features.shape)
    losscolector=[]
    folds=[0,1,2,3,4,5,6,7,8,9]
    seeds=[40,41,42,43,44,45,46,47,48]

    n_fold=len(folds)

    model_dicts=[{'name':'resnetlike','func':Complexer},
                 {'name':'densenetlike','func':Denseplexer}]
    #### 5 fols split
    features = train_features.copy()
    target_cols = [c for c in labels.columns if c not in ['sig_id']]
    features['fold'] = -1
    Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=10086)
    for fold, (train_index, test_index) in enumerate(Fold.split(features, labels[target_cols])):
        features['fold'][test_index] = fold

    for model_dict in model_dicts:
        # for cur_seed in seeds:

        if 1:

            seed_choose_index=0
            for fold in folds:
                
                cur_seed = seeds[seed_choose_index]
                seed_choose_index+=1
                seed_everything(cur_seed)
                logger.info('train with seed %d' % (cur_seed))




                ###build dataset

                train_ind = features[features['fold'] != fold].index.to_list()
                train_features_=features.iloc[train_ind].copy()
                train_target_ = labels.iloc[train_ind].copy()
                train_extra_Target_ = extra_labels.iloc[train_ind].copy()

                val_ind=features.loc[features['fold'] == fold].index.to_list()
                val_features_ = features.iloc[val_ind].copy()
                val_target_ = labels.iloc[val_ind].copy()
                val_extra_Target_ = extra_labels.iloc[val_ind].copy()


                train_ds=DataIter(train_features_,train_target_,train_extra_Target_,shuffle=True,training_flag=True)
                val_ds=DataIter(val_features_,val_target_,val_extra_Target_,shuffle=False,training_flag=False)

                ### build model

                model=model_dict['func']()

                model_name=str(model_dict['name']+str(cur_seed))


                ###build trainer
                trainer = Train(model_name=model_name,model=model,train_ds=train_ds,val_ds=val_ds,fold=fold)

                print('it is here')
                if cfg.TRAIN.vis:
                    print('show it, here')
                    for step in range(train_ds.size):

                        images,data, labels=train_ds()
                        # images, mask, labels = cutmix_numpy(images, mask, labels, 0.5)


                        print(images.shape)

                        for i in range(images.shape[0]):
                            example_image=np.array(images[i,0])

                            example_label=np.array(labels[i])

                            cv2.imshow('ss',example_image)
                            cv2.waitKey(0)



                ### train
                loss,model=trainer.custom_loop()
                losscolector.append([loss,model])

        avg_loss=0
        for k,loss_and_model in enumerate(losscolector):
            print('fold %d : loss %.5f modelname: %s'%(k,loss_and_model[0],loss_and_model[1]))
            avg_loss+=loss_and_model[0]
        print('simple,average loss is ',avg_loss/(len(losscolector)))


if __name__=='__main__':
    main()