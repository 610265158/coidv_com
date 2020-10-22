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
def main():


    feature_file = '../lish-moa/train_features.csv'
    target_file = '../lish-moa/train_targets_scored.csv'
    noscore_target = '../lish-moa/train_targets_nonscored.csv'

    features = pd.read_csv(feature_file)
    labels = pd.read_csv(target_file)
    extra_labels = pd.read_csv(noscore_target)

    test_features = pd.read_csv('../lish-moa/test_features.csv')

    #####FE there

    ####

    print(features.shape)
    losscolector=[]
    folds=[0,1,2,3,4]
    seeds=[40,41,42,43,44]

    n_fold=len(folds)


    for cur_seed in seeds:
        seed_everything(cur_seed)

        #### 5 fols split
        target_cols = [c for c in labels.columns if c not in ['sig_id']]
        features['fold'] = -1
        Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cur_seed)
        for fold, (train_index, test_index) in enumerate(Fold.split(features, labels[target_cols])):
            features['fold'][test_index] = fold


        logger.info('train with seed %d'%(cur_seed))

        for fold in folds:


            ###build dataset

            train_ind = features[features['fold'] != fold].index.to_list()
            train_features=features.iloc[train_ind]
            train_target = labels.iloc[train_ind]
            train_extra_Target = extra_labels.iloc[train_ind]

            val_ind=features.loc[features['fold'] == fold].index.to_list()
            val_features = features.iloc[val_ind]
            val_target = labels.iloc[val_ind]
            val_extra_Target = extra_labels.iloc[val_ind]


            train_ds=DataIter(train_features,train_target,train_extra_Target,shuffle=True,training_flag=True)
            val_ds=DataIter(val_features,val_target,val_extra_Target,shuffle=False,training_flag=False)

            ### build model

            model=Complexer()

            model_name=str('Model2_seed_'+str(cur_seed))
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

    ###blend
    blend_res=[]
    device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    criterion=nn.BCELoss().to(device)
    model = Complexer()

    for k, loss_and_model in enumerate(losscolector):
        print('fold %d : loss %.5f modelname: %s' % (k, loss_and_model[0], loss_and_model[1]))
        avg_loss += loss_and_model[0]
        model.load_state_dict(torch.load(loss_and_model[1], map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            feature, target1, target2 = val_ds()
            feature = torch.from_numpy(feature).to(device).float()
            target1 = torch.from_numpy(target1).to(device).float()
            target2 = torch.from_numpy(target2).to(device).float()

            output,_ = model(feature)
            blend_res.append(torch.nn.functional.sigmoid(output))

    blend_res = torch.stack(blend_res, dim=0)
    blend_res = torch.mean(blend_res, dim=0)

    blend_loss=criterion(blend_res,target1)
    print('blend,average loss is ',blend_loss)


if __name__=='__main__':
    main()