import torch
import torch.nn as nn


class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        return self.loss(output, target).to(dtype=torch.float32)
