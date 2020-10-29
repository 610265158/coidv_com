

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
config.TRAIN.batch_size = 128
config.TRAIN.log_interval = 100                  ##10 iters for a log msg
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 300

config.TRAIN.init_lr=1.e-3

config.TRAIN.weight_decay_factor = 1.e-5                                  ####l2
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
config.MODEL.image_and_data=False
config.MODEL.image_only=False
config.MODEL.pre_length=68         ##68:107, 91:130
config.DATA = edict()

config.DATA.filter_ctl_vehicle=False


####mainly hyper params
config.TRAIN.warmup_step=1500
config.TRAIN.opt='Adamw'
config.TRAIN.SWA=-1    ### -1 use no swa   from which epoch start SWA


config.TRAIN.finetune_alldata=False



config.MODEL.label_smooth=0.05
config.MODEL.cutmix=0.0
config.MODEL.gempool=False

config.MODEL.pretrained_model=None

config.MODEL.freeze_bn=False
config.MODEL.freeze_bn_affine=False

config.MODEL.ema=True
config.MODEL.focal_loss=False
config.SEED=42


from lib.utils import seed_everything

seed_everything(config.SEED)