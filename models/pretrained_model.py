import torch.nn as nn

from torchvision import models
from base import BaseModel


class PretrainedModel(BaseModel):
    def __init__(self, pretrained_model="googlenet", freeze_param=True):
        super(PretrainedModel, self).__init__()
        self.pretrained_model = getattr(models, pretrained_model)(pretrained=True)

        # freeze parameters of pretrained model
        if freeze_param:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # add fully connect layer to output results
        self.fc = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(-1)
         )

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        return x
