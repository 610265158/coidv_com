
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pre, target):
        loss = torch.sqrt(self.mse(pre, target) + self.eps)
        return loss

class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=5):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, pre, target):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(pre[:, :, i], target[:, :, i]) / self.num_scored

        return score