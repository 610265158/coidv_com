

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 1
config.TRAIN.prefetch_size = 15
############

config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 32
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 100

config.TRAIN.init_lr=5.e-4

config.TRAIN.weight_decay_factor = 1.e-2                                  ####l2
config.TRAIN.vis=False                                                      #### if to check the training data


config.TRAIN.vis_mixcut=False
if config.TRAIN.vis:
    config.TRAIN.mix_precision=False                                            ##use mix precision to speedup, tf1.14 at least
else:
    config.TRAIN.mix_precision = False

config.TRAIN.opt='Adamw'

config.MODEL = edict()
config.MODEL.model_path = './models/'                                        ## save directory
config.MODEL.height =  224                                        # input size during training , 128,160,   depends on
config.MODEL.width  =  224

config.MODEL.channel = 3

config.DATA = edict()

config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'

############the model is trained with RGB mode
config.DATA.PIXEL_MEAN = np.array([ 0.460, 0.442 ,0.390 ]).reshape(1,3,1,1)           ###rgb
config.DATA.PIXEL_STD = np.array([0.238, 0.219, 0.232]).reshape(1,3,1,1)





####mainly hyper params
config.TRAIN.warmup_step=-3000
config.TRAIN.opt='Adamw'
config.TRAIN.SWA=-1    ### -1 use no swa   from which epoch start SWA
config.MODEL.label_smooth=0.05
config.MODEL.cutmix=0.0
config.MODEL.gempool=False

config.MODEL.pretrained_model=None

config.MODEL.num_class=9
config.MODEL.freeze_bn=False
config.MODEL.freeze_bn_affine=False

config.MODEL.ema=False
config.MODEL.focal_loss=False
config.SEED=42


from lib.utils import seed_everything

seed_everything(config.SEED)