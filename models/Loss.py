import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLoss, self).__init__()
        self.sigma = nn.Parameter(torch.ones(num_tasks), requires_grad=True)

    def forward(self, *losses):
        losses = torch.cat([loss.unsqueeze(0) for loss in losses])
        loss = (0.5 / torch.pow(self.sigma, 2)) * losses
        return loss.sum() + self.sigma.log().sum()