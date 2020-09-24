
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
    def __init__(self, which_to_score=[0,1,2,3,4]):
        super().__init__()
        self.rmse = RMSELoss()
        self.which_to_score = which_to_score

    def forward(self, pre, target,weights):
        score = 0
        for i in self.which_to_score:
            score += self.rmse(pre[:, :, i], target[:, :, i],weights) / self.num_scored
        return score