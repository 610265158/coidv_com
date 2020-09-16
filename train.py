
from lib.core.base_trainer.net_work import Train
from dataset import train_ds,val_ds
from lib.core.model.ShuffleNet_Series.ShuffleNetV2.network import ShuffleNetV2

from lib.core.model.semodel.SeResnet import se_resnet50
import cv2
import numpy as np

from train_config import config as cfg
import setproctitle

from lib.core.model.mix.mix import cutmix_numpy
setproctitle.setproctitle("alaska")


def main():


    folds=[0]

    for fold in folds:
        ###build dataset

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
        trainer.custom_loop()

if __name__=='__main__':
    main()