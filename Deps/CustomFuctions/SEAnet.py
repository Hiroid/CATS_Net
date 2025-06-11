import torch
import torch.nn as nn
from models import *


class Net1(nn.Module):
    def __init__(self, my_pretrained_classifier, bottle_neck, cdp_hidden, context_dim):
        super(Net1, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.clf_fc1 = nn.Sequential(nn.Linear(512, bottle_neck, bias=False),
                                     nn.ReLU()
                                     )
        self.clf_fc2 = nn.Sequential(nn.Linear(bottle_neck, 2),
                                     )
        self.cdp = nn.Sequential(nn.Linear(context_dim, cdp_hidden, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(cdp_hidden, cdp_hidden, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(cdp_hidden, bottle_neck, bias=False),
                                 nn.Tanh(),
                                )
        self.tanh = nn.Tanh()
    def forward(self, x, contexts):
        x = self.pretrained_clf(x)
        x = x.detach()
        x = self.clf_fc1(x)
        y = self.tanh(contexts)*2.0
        y = self.cdp(y)
        x = x*y
        x = self.clf_fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self, my_pretrained_classifier, my_pretrained_cdp, context_dim, feature_dim=512):
        super(Net2, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.pretrained_cdp = my_pretrained_cdp
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
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
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
    
class Net3(nn.Module):
    def __init__(self, my_pretrained_classifier, my_pretrained_cdp, context_dim, pregating_layers=[200, 200],
                 gating_layers=[100, 100, 100], feature_dim=512, prob_drop=0):
        super(Net3, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.pretrained_cdp = my_pretrained_cdp
        clf_layers = []
        cdp_pregating_layers = []
        cdp_gating_layers = []
        for id in range(len(pregating_layers)):
            if id==0:
                cdp_pregating_layers.append(nn.Sequential(nn.Linear(context_dim, pregating_layers[id]),
                                                          nn.BatchNorm1d(pregating_layers[id]),
                                                          nn.Tanh(),
                                                          nn.Dropout(p=prob_drop)
                                                          )
                                            )
            else:
                cdp_pregating_layers.append(nn.Sequential(nn.Linear(pregating_layers[id-1], pregating_layers[id]),
                                                          nn.BatchNorm1d(pregating_layers[id]),
                                                          nn.Tanh(),
                                                          nn.Dropout(p=prob_drop)
                                                          )
                                            )
        for id in range(len(gating_layers)+1):
            if id==0:
                cdp_gating_layers.append(nn.Sequential(nn.Linear(pregating_layers[len(pregating_layers)-1], feature_dim),
                                                       nn.BatchNorm1d(feature_dim),
                                                       nn.Tanh(),
                                                       nn.Dropout(p=prob_drop)
                                                       )
                                         )
            elif id==1:
                cdp_gating_layers.append(nn.Sequential(nn.Linear(feature_dim, gating_layers[id-1]),
                                                       nn.BatchNorm1d(gating_layers[id - 1]),
                                                       nn.Tanh(),
                                                       nn.Dropout(p=prob_drop)
                                                       )
                                         )
            else:
                cdp_gating_layers.append(nn.Sequential(nn.Linear(gating_layers[id-2], gating_layers[id-1]),
                                                       nn.BatchNorm1d(gating_layers[id - 1]),
                                                       nn.Tanh(),
                                                       nn.Dropout(p=prob_drop)
                                                       )
                                         )
        for id in range(len(gating_layers)+1):
            if id == len(gating_layers):
                clf_layers.append(nn.Linear(gating_layers[id-1], 2))
            elif id==0:
                clf_layers.append(nn.Sequential(nn.Linear(feature_dim, gating_layers[id]),
                                                nn.BatchNorm1d(gating_layers[id]),
                                                nn.Tanh()
                                                )
                                  )
            else:
                clf_layers.append(nn.Sequential(nn.Linear(gating_layers[id-1], gating_layers[id]),
                                                nn.BatchNorm1d(gating_layers[id]),
                                                nn.Tanh())
                                  )
        self.clf = nn.ModuleList(clf_layers)
        self.cdp_pregating = nn.Sequential(*cdp_pregating_layers)
        self.cdp_gating = nn.ModuleList(cdp_gating_layers)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x, contexts):
        x = self.pretrained_clf(x)
        x = x.detach()
        y = self.cdp_pregating(contexts)
        for layer_id, cdp_layer in enumerate(self.cdp_gating):
            if layer_id==0:
                y = cdp_layer(y)
                x = x*y
            else:
                y = cdp_layer(y)
                x = self.clf[layer_id-1](x)
                x = x*y
        x = self.clf[-1](x)
        return x


class Net4(nn.Module):
    def __init__(self, my_pretrained_classifier, my_pretrained_cdp, context_dim, pregating_layers=[200, 200],
                 gating_layers=[100, 100, 100], feature_dim=512, prob_drop=0):
        super(Net4, self).__init__()
        self.pretrained_clf = my_pretrained_classifier
        self.pretrained_cdp = my_pretrained_cdp
        clf_layers = []
        cdp_pregating_layers = []
        cdp_gating_layers = []
        for id in range(len(pregating_layers)):
            if id==0:
                cdp_pregating_layers.append(nn.Sequential(nn.Linear(context_dim, pregating_layers[id]),
                                                          nn.BatchNorm1d(pregating_layers[id]),
                                                          nn.Sigmoid(),
                                                          nn.Dropout(p=prob_drop)
                                                          )
                                            )
            else:
                cdp_pregating_layers.append(nn.Sequential(nn.Linear(pregating_layers[id-1], pregating_layers[id]),
                                                          nn.BatchNorm1d(pregating_layers[id]),
                                                          nn.Sigmoid(),
                                                          nn.Dropout(p=prob_drop)
                                                          )
                                            )
        for id in range(len(gating_layers)+1):
            if id==0:
                cdp_gating_layers.append(nn.Sequential(nn.Linear(pregating_layers[len(pregating_layers)-1], feature_dim),
                                                       nn.BatchNorm1d(feature_dim),
                                                       nn.Sigmoid(),
                                                       nn.Dropout(p=prob_drop)
                                                       )
                                         )
            elif id==1:
                cdp_gating_layers.append(nn.Sequential(nn.Linear(feature_dim, gating_layers[id-1]),
                                                       nn.BatchNorm1d(gating_layers[id - 1]),
                                                       nn.Sigmoid(),
                                                       nn.Dropout(p=prob_drop)
                                                       )
                                         )
            else:
                cdp_gating_layers.append(nn.Sequential(nn.Linear(gating_layers[id-2], gating_layers[id-1]),
                                                       nn.BatchNorm1d(gating_layers[id - 1]),
                                                       nn.Sigmoid(),
                                                       nn.Dropout(p=prob_drop)
                                                       )
                                         )
        for id in range(len(gating_layers)+1):
            if id == len(gating_layers):
                clf_layers.append(nn.Linear(gating_layers[id-1], 2))
            elif id==0:
                clf_layers.append(nn.Sequential(nn.Linear(feature_dim, gating_layers[id]),
                                                nn.BatchNorm1d(gating_layers[id]),
                                                nn.ReLU()
                                                )
                                  )
            else:
                clf_layers.append(nn.Sequential(nn.Linear(gating_layers[id-1], gating_layers[id]),
                                                nn.BatchNorm1d(gating_layers[id]),
                                                nn.ReLU())
                                  )
        self.clf = nn.ModuleList(clf_layers)
        self.cdp_pregating = nn.Sequential(*cdp_pregating_layers)
        self.cdp_gating = nn.ModuleList(cdp_gating_layers)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x, contexts):
        x = self.pretrained_clf(x)
        x = x
        y = self.cdp_pregating(contexts)
        for layer_id, cdp_layer in enumerate(self.cdp_gating):
            if layer_id==0:
                y = cdp_layer(y)
                x = x*y
            else:
                y = cdp_layer(y)
                x = self.clf[layer_id-1](x)
                x = x*y
        x = self.clf[-1](x)
        return x