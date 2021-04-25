import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        untrained_parameters = filter(lambda p: not p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        self.un_params = sum([np.prod(p.size()) for p in untrained_parameters])
        return f"{super().__str__()} \nTrainable parameters: {self.params}\nFixed parameters: {self.un_params}"
