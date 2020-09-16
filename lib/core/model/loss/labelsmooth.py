import torch
import torch.nn as nn

from torch.autograd import Variable

from train_config import config as cfg

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing


    def forward(self, x, target,mean=True):


        N = x.size(0)
        C = x.size(1)
        class_mask = x.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target=class_mask
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target

            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = (self.confidence * nll_loss + self.smoothing * smooth_loss)

            if mean:
                return loss.mean()
            else:
                return loss
        else:
            return torch.nn.functional.cross_entropy(x, target)