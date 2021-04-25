import torch.nn as nn

from torchvision import models
from base import BaseModel


class Resnet(BaseModel):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = models.googlenet(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(-1)
         )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
