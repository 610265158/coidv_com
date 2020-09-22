
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
from lib.core.base_trainer.model import GRU_model,Complexer
from functools import partial
def main():




    data = pd.read_json('folds.json',lines=True)



    losscolector=[]
    folds=[0,1,2,3,4]

    models=[{'model_name':'gru','model':Complexer,'mtype':0,"weights":[]},
            {'model_name': 'lstm', 'model': Complexer, 'mtype': 1,"weights":[]}
            ]
    for model_type in models:
        for fold in folds:

            ###build dataset
            train_data=data[data['fold']!=fold]
            val_data=data[data['fold']==fold]

            train_ds=DataIter(train_data,shuffle=True,training_flag=True)
            val_ds=DataIter(val_data,shuffle=False,training_flag=False)
            ###build trainer


            trainer = Train(model=model_type['model'](mtype=model_type['mtype']),
                            model_name=model_type['model_name'],
                            train_ds=train_ds,val_ds=val_ds,fold=fold)

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
            model_type['weights'].append(model)
            losscolector.append([loss,model])

        avg_loss=0
        for k,loss_and_model in enumerate(losscolector):
            print('model_name %s. fold %d : loss %.5f modelname: %s'%( model_type['model_name'],k,loss_and_model[0],loss_and_model[1]))
            avg_loss+=loss_and_model[0]
        print('average loss is ',avg_loss/len(folds))


    print('final sub\n',model_type)
if __name__=='__main__':
    main()