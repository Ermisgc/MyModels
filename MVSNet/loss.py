import torch
import torch.nn as nn
import torch.nn.functional as F


def mvsnet_loss(gt, initial_depth, refined_depth, lamb=1.0):
    return F.smooth_l1_loss(gt, initial_depth, size_average=True) + lamb * F.smooth_l1_loss(gt, refined_depth, size_average=True)
