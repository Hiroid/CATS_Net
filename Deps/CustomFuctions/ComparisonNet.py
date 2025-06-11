import torch
import torch.nn as nn
from models import *

class CNNNet(nn.Module):
    def __init__(self, my_pretrained_classifier, output_neuron_num):
        super(CNNNet, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.clf_fc1 = nn.Sequential(nn.Linear(512, 500, bias=False),
                                     nn.BatchNorm1d(500),
                                     nn.ReLU()
                                     )
        self.clf_fc2 = nn.Sequential(nn.Linear(500, 500, bias=False),
                                     nn.BatchNorm1d(500),
                                     nn.ReLU()
                                     )
        self.clf_fc3 = nn.Sequential(nn.Linear(500, output_neuron_num),
                                     )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pretrained_clf(x)

        x = self.clf_fc1(x)
        x = self.clf_fc2(x)
        x = self.clf_fc3(x)
        return x

class CNNNet_detach(nn.Module):
    def __init__(self, my_pretrained_classifier, output_neuron_num):
        super(CNNNet_detach, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.clf_fc1 = nn.Sequential(nn.Linear(512, 500, bias=False),
                                     nn.BatchNorm1d(500),
                                     nn.ReLU()
                                     )
        self.clf_fc2 = nn.Sequential(nn.Linear(500, 500, bias=False),
                                     nn.BatchNorm1d(500),
                                     nn.ReLU()
                                     )
        self.clf_fc3 = nn.Sequential(nn.Linear(500, output_neuron_num),
                                     )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pretrained_clf(x)
        x = x.detach()
        x = self.clf_fc1(x)
        x = self.clf_fc2(x)
        x = self.clf_fc3(x)
        return x