

import torch


import torch.nn.functional as F

import torch.nn as nn




from lib.core.model.loss.labelsmooth import LabelSmoothing


from  train_config import config as cfg



class OHEMLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, batch_size,ratio=0.5 ):
        super(OHEMLoss, self).__init__()


        self.batch_size = batch_size
        self.top_k=int(self.batch_size*ratio)

        self.losser=LabelSmoothing(smoothing=cfg.MODEL.label_smooth)
    def forward(self, cls_pred, cls_target, ratio=0.5):


        ### refine topk
        self.top_k = int(self.batch_size * ratio)

        ohem_cls_loss =self.losser(cls_pred, cls_target,False)

        # 这里先暂存下正常的分类loss和回归loss
        loss = ohem_cls_loss
        # 然后对分类和回归loss求和

        sorted_ohem_loss, idx = torch.sort(loss, descending=True)
        # 再对loss进行降序排列
        keep_num = min(sorted_ohem_loss.size()[0], self.top_k)
        # 得到需要保留的loss数量
        if keep_num < sorted_ohem_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
            keep_idx_cuda = idx[:keep_num]
            # 保留到需要keep的数目
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]

            # 分类和回归保留相同的数目
        cls_loss = ohem_cls_loss.sum() / keep_num

        # 然后分别对分类和回归loss求均值
        return cls_loss

