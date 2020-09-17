
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


def main():




    data = pd.read_json('folds.json',lines=True)



    losscolector=[]
    folds=[0,1,2,3,4]

    for fold in folds:

        ###build dataset
        train_data=data[data['fold']!=fold]
        val_data=data[data['fold']==fold]

        train_ds=DataIter(train_data,shuffle=True,training_flag=True)
        val_ds=DataIter(val_data,shuffle=False,training_flag=False)
        ###build trainer
        trainer = Train(train_ds=train_ds,val_ds=val_ds,fold=fold)

        print('it is here')
        if cfg.TRAIN.vis:
            print('show it, here')
            for step in range(train_ds.size):

                images, labels=train_ds()
                # images, mask, labels = cutmix_numpy(images, mask, labels, 0.5)


                print(images.shape)

                for i in range(images.shape[0]):
                    example_image=np.array(images[i])

                    example_label=np.array(labels[i])

                    break


                break

        ### train
        loss=trainer.custom_loop()
        losscolector.append(loss)

    avg_loss=0
    for k,loss in enumerate(losscolector):
        print('fold %d : loss %.5f'%(k,loss))
        avg_loss+=loss
    print('average loss is ',avg_loss/5.)


if __name__=='__main__':
    main()