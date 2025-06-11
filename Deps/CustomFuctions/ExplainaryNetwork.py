import torch
import torch.nn as nn
from models import *


class Net1(nn.Module):
    def __init__(self, input_dim, num_class, bottle_neck, fc_layer_num, prob_drop):
        super(Net1, self).__init__()
        fc_layer = [nn.Linear(input_dim, bottle_neck), nn.ReLU(), nn.Dropout(p=prob_drop)]
        for i in range(fc_layer_num):
            fc_layer.append(nn.Linear(bottle_neck, bottle_neck))
            fc_layer.append(nn.ReLU())
            fc_layer.append(nn.Dropout(p=prob_drop))
        self.fc = nn.ModuleList(fc_layer)
        self.predictor = nn.Linear(bottle_neck, num_class)
        self.tanh = nn.Tanh()
    def forward(self, x):
        for m in self.fc:
            x = m(x)
        x = self.predictor(x)
        return x