

import random
import cv2
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator,BatchData, MultiProcessPrefetchData
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
    def __init__(self,data,training_flag=True,shuffle=True):

        self.shuffle=shuffle
        self.training_flag=training_flag
        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size


        if not training_flag:
            self.process_num=1
        self.generator = AlaskaDataIter(data, self.training_flag,self.shuffle)

        self.ds=self.build_iter()

        self.size = self.__len__()


    def parse_file(self,im_root_path,ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")


    def build_iter(self):

        ds = DataFromGenerator(self.generator)
        ds = BatchData(ds, self.batch_size)
        if not cfg.TRAIN.vis:
            ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __iter__(self):

        for i in range(self.size):
            one_batch = next(self.ds)

            yield one_batch[0], one_batch[1]

    def __call__(self, *args, **kwargs):





        one_batch=next(self.ds)

        image,data,label=one_batch[0],one_batch[1],one_batch[2]

        return image,data,label



    def __len__(self):
        return len(self.generator)//self.batch_size

    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class AlaskaDataIter():
    def __init__(self,data, training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle
        self.SNR_THRS=1.


        iid,data,label=self.parse_file(data)



        self.iid=iid
        self.data=data
        self.label=label
        self.raw_data_set_size = self.data.shape[0]  ##decided by self.parse_file

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


    def parse_file(self,train):
        # target columns
        target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}

        rna_type={x: i for i, x in enumerate('ACGU')}
        struct_type = {x: i for i, x in enumerate('().')}
        loop_type= {x: i for i, x in enumerate('BEHIMSX')}

        def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

            seq=np.array(df[['sequence']]
                              .applymap(lambda seq: [rna_type[x] for x in seq])
                              .values
                              .tolist()
                              )
            struc=np.array(df[['structure']]
                              .applymap(lambda seq: [struct_type[x] for x in seq])
                              .values
                              .tolist()
                              )
            loop = np.array(df[['predicted_loop_type']]
                             .applymap(lambda seq: [loop_type[x] for x in seq])
                             .values
                             .tolist()
                             )

            encode=np.concatenate([seq,struc,loop],axis=1)

            return encode

        train_inputs = preprocess_inputs(train[train.signal_to_noise > self.SNR_THRS])
        train_labels = np.array(train[train.signal_to_noise > self.SNR_THRS][target_cols].values.tolist())

        train_id=train[train.signal_to_noise > self.SNR_THRS]['id'].values.tolist()

        logger.info('contains %d samples'%(train_labels.shape[0]) )

        return train_id,train_inputs,train_labels

    def onehot(self,lable,depth=1000):
        length=lable.shape[0]
        one_hot_label=np.zeros(shape=[length,depth])

        for i in range(length):
            one_hot_label[i][lable[i]]=1
        return one_hot_label

    def get_one_sample(self,id,training):
        iid = self.iid[id]

        bpp_path = os.path.join('../stanford-covid-vaccine/bpps', iid + '.npy')

        image = np.load(bpp_path)
        image = np.expand_dims(image, axis=0)
        data = self.data[id]
        label = self.label[id]

        data = np.transpose(data, [1, 0])  ##shape [n,107,3)
        label = np.transpose(label, [1, 0])


        return image, data, label


    def pad_to_long(self,data,label,length=130,extra_length=23,training=True):

        ##better simulate to private dataset


        random_iid=random.randint(0,len(self.iid)-1)
        random_image, random_data, random_label=self.get_one_sample(random_iid,training)

        start=random.randint(0,68-extra_length)

        end=start+extra_length
        cropped_data=random_data[start:end,:]
        cropped_label = random_label[start:end, :]

        ####join
        mid=0

        left_data=data[0:mid,:]
        right_data=data[mid:,:]
        left_label = label[0:mid, :]
        right_label = label[mid:, :]

        data=np.concatenate([left_data,cropped_data,right_data])
        label=np.concatenate([left_label,cropped_label,right_label])

        return data,label

    def single_map_func(self, id, is_training):
        """Data augmentation function."""
        ####customed here

        image,data,label=self.get_one_sample(id,is_training)

        if cfg.MODEL.pre_length==91:
            data, label=self.pad_to_long(data,label)


        seq=data[:,0]
        strc=data[:,1]
        loop=data[:,2]

        seq=self.onehot(seq,4)
        strc = self.onehot(strc,3)
        loop = self.onehot(loop,7)

        data=np.concatenate([seq,strc,loop],axis=1)


        # if is_training:
        #
        #     if random.uniform(0,1)<0.5:
        #
        #
        #         data[:cfg.MODEL.pre_length,:]=data[:cfg.MODEL.pre_length][::-1,:]
        #         label[:cfg.MODEL.pre_length,:]=label[:cfg.MODEL.pre_length][::-1,:]

        return image,data,label
