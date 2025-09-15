import torch
import torch.nn as nn
from models import *

class Net2(nn.Module):
    def __init__(self, my_pretrained_classifier, context_dim, feature_dim=512):
        super(Net2, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.clf_fc1 = nn.Sequential(nn.Linear(feature_dim, 100, bias=False),
                                     nn.BatchNorm1d(100),
                                     nn.ReLU()
                                     )
        self.clf_fc2 = nn.Sequential(nn.Linear(100, 10, bias=False),
                                     nn.BatchNorm1d(10),
                                     nn.ReLU()
                                     )
        self.clf_fc3 = nn.Sequential(nn.Linear(10, 2),
                                     )
        self.cdp_fc1 = nn.Sequential(nn.Linear(context_dim, feature_dim, bias=False),
                                     nn.BatchNorm1d(feature_dim),
                                     nn.Sigmoid()
                                     )
        self.cdp_fc2 = nn.Sequential(nn.Linear(feature_dim, 100, bias=False),
                                     nn.BatchNorm1d(100),
                                     nn.Sigmoid()
                                     )
        self.cdp_fc3 = nn.Sequential(nn.Linear(100, 10, bias=False),
                                     nn.BatchNorm1d(10),
                                     nn.Sigmoid()
                                     )

    def forward(self, x, contexts):
        x = self.pretrained_clf(x)
        x = x.detach()
        y = self.cdp_fc1(contexts)
        x = x*y
        x = self.clf_fc1(x)
        y = self.cdp_fc2(y)
        x = x*y
        x = self.clf_fc2(x)
        y = self.cdp_fc3(y)
        x = x*y
        x = self.clf_fc3(x)
        return x