import torch
import torch.nn as nn
from models import *

class Net(nn.Module):
    def __init__(self, my_pretrained_classifier, bottle_neck, fc_layer_num, prob_drop, context_dim, feature_num=512):
        super(Net, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        fc_layer = [nn.Linear(feature_num, bottle_neck), nn.ReLU(), nn.Dropout(p=prob_drop)]
        for i in range(fc_layer_num):
            fc_layer.append(nn.Linear(bottle_neck, bottle_neck))
            fc_layer.append(nn.ReLU())
            fc_layer.append(nn.Dropout(p=prob_drop))
        self.fc = nn.ModuleList(fc_layer)
        self.predictor = nn.Linear(bottle_neck, context_dim)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.pretrained_clf(x)
        for m in self.fc:
            x = m(x)
        x = self.predictor(x)
        return x