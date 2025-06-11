import torch
import torch.nn as nn
import numpy as np

def Cifar100_fine2coarse(labels):
    class_interval = 5
    coarse_label = labels/class_interval
    coarse_labels = torch.tensor(coarse_label.floor(), dtype=torch.long)
    return coarse_labels