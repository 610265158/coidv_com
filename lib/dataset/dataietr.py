
import pandas as pd
import random
import cv2
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator, BatchData, MultiProcessPrefetchData, PrefetchDataZMQ, RepeatedData
import time


from lib.dataset.augmentor.augmentation import Rotate_aug,\
                                                Affine_aug,\
                                                Mirror,\
                                                Padding_aug,\
                                                Img_dropout



from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from train_config import config as cfg
import albumentations as A
import os

class data_info(object):
    def __init__(self,img_root,ann_file,training=True):
        self.ann_file=ann_file
        self.root_path = img_root
        self.metas=[]
        self.training=training

        self.load_anns()
    def one_hot(self,p,length):
        label=np.zeros(shape=length)
        label[p]=1
        return label

    def load_anns(self):
        with open(self.ann_file, 'r') as f:
            image_label_list = f.readlines()

        for line in image_label_list:
            cur_data_info = line.rstrip().split('|')
            fname = cur_data_info[0]
            label = cur_data_info[1]




            image_path= os.path.join(self.root_path,fname)
            self.metas.append([image_path, label])

            ###some change can be made here

        logger.info('the datasets contains %d samples'%(len(image_label_list)))
        logger.info('the datasets contains %d samples after filter' % (len(self.metas)))

    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas
#
#
class DataIter():
    def __init__(self,feature,target,extra_target,training_flag=True,shuffle=True):

        self.shuffle=shuffle
        self.training_flag=training_flag
        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size



        self.generator = AlaskaDataIter(feature,target,extra_target, self.training_flag,self.shuffle)
        if not training_flag:
            self.process_num=1
            self.batch_size=len(self.generator)

        self.ds=self.build_iter()

        self.size = self.__len__()


    def parse_file(self,im_root_path,ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")


    def build_iter(self):

        ds = DataFromGenerator(self.generator)
        ds = RepeatedData(ds, -1)
        ds = BatchData(ds, self.batch_size)
        if not cfg.TRAIN.vis:
            ds = PrefetchDataZMQ(ds, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __call__(self, *args, **kwargs):


        one_batch=next(self.ds)

        image,label1,label2=one_batch[0],one_batch[1],one_batch[2]

        return image,label1,label2



    def __len__(self):
        if not self.training_flag:
            return 1
        else:
            return len(self.generator)//self.batch_size

    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class AlaskaDataIter():
    def __init__(self,feature,target,extra_target, training_flag=True,shuffle=True):



        ###prevent modify
        feature=feature.copy()

        self.training_flag = training_flag
        self.shuffle = shuffle

        data,target,extra_target=self.parse_file(feature,target,extra_target)




        self.data=data
        self.label=target
        self.extra_label=extra_target
        self.raw_data_set_size = self.data.shape[0]  ##decided by self.parse_file



        self.get_the_thres(self.data,target)

    def __call__(self, *args, **kwargs):
        idxs = np.arange(self.data.shape[0])


        while 1:
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in idxs:
                yield self.single_map_func(k, self.training_flag)


    def __len__(self):
        assert self.raw_data_set_size is not None

        return self.raw_data_set_size




    def get_the_thres(self,feature,target):

        take = np.sum(target, axis=1)

        posotive_index = (take > 0)

        data = feature[posotive_index]
        self.pos_max = np.max(data, axis=0)
        self.pos_min = np.min(data, axis=0)

        neg_index = (take == 0)

        data = feature[neg_index]
        self.neg_max = np.max(data, axis=0)
        self.neg_min = np.min(data, axis=0)

    def parse_file(self,feature,target,extra_target):




        train_features = feature
        labels_train = target
        extra_labels_train = extra_target

        def preprocess(df):
            """Returns preprocessed data frame"""
            df = df.copy()
            df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
            df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
            df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})

            return df

        train_features=preprocess(train_features)

        ####filter control
        if cfg.DATA.filter_ctl_vehicle:
            filter_index = train_features['cp_type'] != 1
            train_features = train_features[filter_index]
            labels_train = labels_train[filter_index]
            extra_labels_train = extra_labels_train[filter_index]

        train_features = train_features.drop(['sig_id', 'fold' ], axis=1).values

        labels_train = labels_train.drop('sig_id', axis=1).values
        extra_labels_train = extra_labels_train.drop('sig_id', axis=1).values


        logger.info('dataset contains %d samples'%(train_features.shape[0]))

        return train_features,labels_train,extra_labels_train

    def single_map_func(self, index, is_training):
        """Data augmentation function."""
        ####customed here
        data = self.data[index].copy()

        target = self.label[index]
        extra_target = self.extra_label[index]


        if is_training:
            if random.uniform(0,1)<0.5:
                data[3:]=self.jitter(data[3:])
            if random.uniform(0,1)<0.5:
                data[3:]=self.cutout(data[3:])

            if np.sum(target)>0:
                data[3:]=np.clip(data[3:],self.pos_min[3:],self.pos_max[3:])
            else:
                data[3:] = np.clip(data[3:], self.neg_min[3:], self.neg_max[3:])
                
        return data,target,extra_target



    def jitter(self,x, rate=0.5):

        mask=np.random.uniform(0,1,size=x.shape[0])

        mask=mask>rate

        jitter=np.random.uniform(-1,1,size=x.shape[0])*mask*2

        return x+jitter
    def cutout(self,x, rate=0.2):

        mask=np.random.uniform(0,1,size=x.shape[0])

        mask=mask>rate

        return x*mask


