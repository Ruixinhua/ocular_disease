# -*- coding: utf-8 -*-
# The helper function for loss calculation
import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, log_var):
    mse = F.mse_loss(recon_x, x)
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return mse + kld


def cross_loss(pred, label):
    return nn.CrossEntropyLoss()(pred, label)
