from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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

from lib.core.base_trainer.model import Complexer
def main():





    feature_file = '../lish-moa/train_features.csv'
    target_file = '../lish-moa/train_targets_scored.csv'
    noscore_target = '../lish-moa/train_targets_nonscored.csv'

    features = pd.read_csv(feature_file)
    labels = pd.read_csv(target_file)
    extra_labels = pd.read_csv(noscore_target)



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
    print('average loss is ',avg_loss/(len(losscolector)))


if __name__=='__main__':
    main()