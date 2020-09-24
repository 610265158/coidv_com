
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, pre, target,weights):
        weights=weights.unsqueeze(-1)
        loss = torch.sqrt(torch.mean(self.mse(pre, target)*weights + self.eps))
        return loss

class MCRMSELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rmse = RMSELoss()


    def forward(self, pre, target,weights,which_to_score=[0,1,2,3,4]):
        score = 0
        for i in which_to_score:
            score += self.rmse(pre[:, :, i], target[:, :, i],weights) / len(which_to_score)
        return score