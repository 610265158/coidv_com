

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
    def __init__(self,data,augdata,training_flag=True,shuffle=True):

        self.shuffle=shuffle
        self.training_flag=training_flag
        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size



        self.generator = AlaskaDataIter(data,augdata, self.training_flag,self.shuffle)
        if not training_flag:
            self.process_num=1
            self.batch_size=len(self.generator)

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
        if not self.training_flag:
            return 1
        else:
            return len(self.generator)//self.batch_size

    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class AlaskaDataIter():
    def __init__(self,data,augdata, training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle
        self.SNR_THRS=1.

        if cfg.DATA.AUG and augdata is not  None:

            data=self.aug_data(data,augdata)


        raw_data,data,label=self.parse_file(data)



        self.raw_data=raw_data
        self.data=data
        self.label=label
        self.raw_data_set_size = self.data.shape[0]  ##decided by self.parse_file


    def aug_data(self,df,aug_df,filter_noise=True):
        if filter_noise:
            df=df[df.signal_to_noise >cfg.DATA.filter_noise]
        target_df = df.copy()
        new_df = aug_df[aug_df['id'].isin(target_df['id'])]

        del target_df['structure']
        del target_df['predicted_loop_type']
        new_df = new_df.merge(target_df, on=['id', 'sequence'], how='left')

        df['cnt'] = df['id'].map(new_df[['id', 'cnt']].set_index('id').to_dict()['cnt'])
        df['log_gamma'] = 100
        df['score'] = 1.0
        df = df.append(new_df[df.columns])
        return df

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

        def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
            encode = np.array(df[cols]
                              .applymap(lambda seq: [token2int[x] for x in seq])
                              .values
                              .tolist()
                              )

            bpps_max=[]
            bpps_sum = []
            bpps_np=[]

            for mol_id in df.id.to_list():

                image=np.load(f"../stanford-covid-vaccine/bpps/{mol_id}.npy")

                bpp_max = np.max(image, axis=-1)
                bpp_sum = np.sum(image, axis=-1)

                # bpp_nb_mean = 0.077522  # mean of bpps_nb across all training data
                # bpp_nb_std = 0.08914  # std of bpps_nb across all training data
                bpp_nb = (image > 0).sum(axis=0) / image.shape[0]
                #bpp_nb = (bpp_nb - bpp_nb_mean) / bpp_nb_std

                bpps_max.append(bpp_max)
                bpps_sum.append(bpp_sum)
                bpps_np.append(bpp_nb)

            bpps_max= np.expand_dims(np.array(bpps_max),1)
            bpps_sum = np.expand_dims(np.array(bpps_sum),1)
            bpps_np = np.expand_dims(np.array(bpps_np),1)

            data = np.concatenate([encode,bpps_max,bpps_sum,bpps_np],axis=1)


            return data

        if not self.training_flag:
            train_inputs = preprocess_inputs(train[train.signal_to_noise > self.SNR_THRS])
            train_labels = np.array(train[train.signal_to_noise > self.SNR_THRS][target_cols].values.tolist())

            train=train[train.signal_to_noise > self.SNR_THRS]

        else:
            train_inputs = preprocess_inputs(train)
            train_labels = np.array(train[target_cols].values.tolist())


        logger.info('contains %d samples'%(train_labels.shape[0]) )

        return train,train_inputs,train_labels

    def onehot(self,lable,depth=1000):
        length=lable.shape[0]
        one_hot_label=np.zeros(shape=[length,depth])

        for i in range(length):
            one_hot_label[i][lable[i]]=1
        return one_hot_label


    def get_one_sample(self,index,training):

        #iid = self.raw_data.iloc[index]['id']

        snr=self.raw_data.iloc[index]['signal_to_noise']

        if training:
            weights=np.log(snr + 1.1) / 2
        else:
            weights = 1

        data = self.data[index]
        label = self.label[index]

        data = np.transpose(data, [1,0])  ##shape [n,107,3)
        label = np.transpose(label, [1,0])

        return data, label,weights


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

    def single_map_func(self, index, is_training):
        """Data augmentation function."""
        ####customed here

        data,label,weights=self.get_one_sample(index,is_training)

        if cfg.MODEL.pre_length==91:
            data, label=self.pad_to_long(data,label)

        # if is_training:
        #
        #     if random.uniform(0,1)<0.5:
        #
        #
        #         data[:cfg.MODEL.pre_length,:]=data[:cfg.MODEL.pre_length][::-1,:]
        #         label[:cfg.MODEL.pre_length,:]=label[:cfg.MODEL.pre_length][::-1,:]

        return data,label,weights