#-*-coding:utf-8-*-

import sklearn.metrics
import cv2
import time
import os


from lib.core.utils.torch_utils import EMA
from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

import sklearn.metrics
from lib.helper.logger import logger

from lib.core.model.ShuffleNet_Series.ShuffleNetV2.utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from lib.core.model.loss.focal_loss import FocalLoss,FocalLoss4d
from lib.core.base_trainer.model import Complexer


from lib.core.base_trainer.metric import *
import torch

from lib.core.base_trainer.loss import BCEWithLogitsLoss
from torchcontrib.optim import SWA





if cfg.TRAIN.mix_precision:
    from apex import amp



class Train(object):
  """Train class.
  """

  def __init__(self,model_name,model,train_ds,val_ds,fold):


    self.model_name=model_name
    self.fold=fold

    self.init_lr=cfg.TRAIN.init_lr
    self.warup_step=cfg.TRAIN.warmup_step
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor

    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    self.model = model.to(self.device)

    self.load_weight()

    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.TRAIN.weight_decay_factor},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if 'Adamw' in cfg.TRAIN.opt:

      self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.init_lr,eps=1.e-5)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=self.init_lr,
                                       momentum=0.9)

    if cfg.TRAIN.SWA>0:
        ##use swa
        self.optimizer = SWA(self.optimizer)

    if cfg.TRAIN.mix_precision:
        self.model, self.optimizer = amp.initialize( self.model, self.optimizer, opt_level="O1")

    if cfg.TRAIN.num_gpu>1:
        self.model=nn.DataParallel(self.model)

    self.ema = EMA(self.model, 0.99)

    self.ema.register()
    ###control vars
    self.iter_num=0

    self.train_ds=train_ds

    self.val_ds = val_ds

    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', patience=5,min_lr=1e-6,verbose=True)
    # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, self.epochs,eta_min=1.e-6)



    self.train_criterion=BCEWithLogitsLoss(smooth_eps=0.001).to(self.device)
    self.criterion = nn.BCEWithLogitsLoss().to(self.device)

  def custom_loop(self):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(epoch_num):

      summary_loss = AverageMeter()

      self.model.train()

      if cfg.MODEL.freeze_bn:
          for m in self.model.modules():
              if isinstance(m, nn.BatchNorm2d):
                  m.eval()
                  if cfg.MODEL.freeze_bn_affine:
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False
      for step in range(self.train_ds.size):

        if epoch_num<10:
            ###excute warm up in the first epoch
            if self.warup_step>0:
                if self.iter_num < self.warup_step:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                        lr = param_group['lr']

                    logger.info('warm up with learning rate: [%f]' % (lr))

        start=time.time()

        feature, target1, target2 = self.train_ds()
        feature = torch.from_numpy(feature).to(self.device).float()
        target1 = torch.from_numpy(target1).to(self.device).float()
        target2 = torch.from_numpy(target2).to(self.device).float()
        batch_size = feature.shape[0]

        output,output2 = self.model(feature)
        loss1=self.train_criterion(output,target1)
        loss2 = self.train_criterion(output2, target2)
        loss=loss1+loss2
        summary_loss.update(loss.detach().item(), batch_size)

        self.optimizer.zero_grad()

        if cfg.TRAIN.mix_precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.optimizer.step()
        if cfg.MODEL.ema:
            self.ema.update()
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:

            log_message = '[fold %d], '\
                          'Train Step %d, ' \
                          'summary_loss: %.6f, ' \
                          'time: %.6f, '\
                          'speed %d images/persec'% (
                              self.fold,
                              self.iter_num,
                              summary_loss.avg,
                              time.time() - start,
                              images_per_sec)
            logger.info(log_message)



      if cfg.TRAIN.SWA>0 and epoch_num>=cfg.TRAIN.SWA:
        self.optimizer.update_swa()

      return summary_loss
    def distributed_test_epoch(epoch_num):
        summary_loss = AverageMeter()


        self.model.eval()
        t = time.time()
        with torch.no_grad():
            for step in range(self.val_ds.size):
                feature, target1, target2 = self.val_ds()
                feature = torch.from_numpy(feature).to(self.device).float()
                target1 = torch.from_numpy(target1).to(self.device).float()
                target2 = torch.from_numpy(target2).to(self.device).float()
                batch_size = feature.shape[0]

                output,_ = self.model(feature)
                loss = self.criterion(output, target1)
                summary_loss.update(loss.detach().item(), batch_size)

                if step % cfg.TRAIN.log_interval == 0:

                    log_message = '[fold %d], '\
                                  'Val Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'time: %.6f' % (
                                  self.fold,step, summary_loss.avg, time.time() - t)

                    logger.info(log_message)


        return summary_loss

    best_loss = 10000.
    best_model = 'xxx'
    not_improvement = 0

    for epoch in range(self.epochs):



      for param_group in self.optimizer.param_groups:
        lr=param_group['lr']
      logger.info('learning rate: [%f]' %(lr))
      t=time.time()

      summary_loss = distributed_train_epoch(epoch)

      train_epoch_log_message = '[fold %d], '\
                                '[RESULT]: Train. Epoch: %d,' \
                                ' summary_loss: %.5f,' \
                                ' time:%.5f' % (
                                self.fold,epoch, summary_loss.avg, (time.time() - t))
      logger.info(train_epoch_log_message)

      if cfg.TRAIN.SWA > 0 and epoch >=cfg.TRAIN.SWA:

          ###switch to avg model
          self.optimizer.swap_swa_sgd()


      ##switch eam weighta
      if cfg.MODEL.ema:
        self.ema.apply_shadow()

      if epoch%cfg.TRAIN.test_interval==0:

          summary_loss = distributed_test_epoch(epoch)

          val_epoch_log_message = '[fold %d], '\
                                  '[RESULT]: VAL. Epoch: %d,' \
                                  ' summary_loss: %.5f,' \
                                  ' time:%.5f' % (
                                   self.fold,epoch, summary_loss.avg,(time.time() - t))
          logger.info(val_epoch_log_message)

      # self.scheduler.step()
      self.scheduler.step(summary_loss.avg)


      #### save model
      if not os.access(cfg.MODEL.model_path, os.F_OK):
          os.mkdir(cfg.MODEL.model_path)
      ###save the best auc model


      if summary_loss.avg<best_loss:

          if os.access(best_model, os.F_OK):
              os.remove(best_model)

          #### save the model every end of epoch
          current_model_saved_name = './models/%s_fold%d_epoch_%d_val_loss%.6f.pth' % (self.model_name,self.fold, epoch, summary_loss.avg)

          logger.info('A model saved to %s' % current_model_saved_name)
          torch.save(self.model.state_dict(), current_model_saved_name)

          best_loss=summary_loss.avg
          best_model=current_model_saved_name
          not_improvement=0

      else:
          not_improvement+=1

      if not_improvement>20:

          logger.info('val loss not improve for 20 epochs ,was stopped, with best loss model %s'%(best_model))
          break

      ####switch back
      if cfg.MODEL.ema:
        self.ema.restore()

      if cfg.TRAIN.SWA > 0 and epoch > cfg.TRAIN.SWA:
          ###switch back to plain model to train next epoch
          self.optimizer.swap_swa_sgd()


    return best_loss,best_model

  def load_weight(self):
      if cfg.MODEL.pretrained_model is not None:
          state_dict=torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
          self.model.load_state_dict(state_dict,strict=False)



