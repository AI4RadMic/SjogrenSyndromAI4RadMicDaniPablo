from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class TL_ResNet(nn.Module):
    def __init__(self, num_classes = 4, dropout_rate=0.0, device='cpu'):
        super(TL_ResNet, self).__init__()
        self.model_ft = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

        self.softmax = nn.Softmax(dim = 1)

        self.device = torch.device(device)
        self.to(self.device)
        

    def forward(self, x):
        x = self.model_ft(x.repeat(1, 3, 1, 1))
        return x
