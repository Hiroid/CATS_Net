from turtle import forward
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torchvision
from torchvision.models import resnet18, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from . import utils
from pathlib import Path
import os

script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent

class cats_net(nn.Module):
    def __init__(self, symbol_size, num_new = 12, num_classes = 10, fix_fe = True, fe_type = 'resnet18', pretrain = False):
        super().__init__()
        self.fix_fe = fix_fe
        self.fe = torchvision.models.__dict__[fe_type](weights=None)
        self.fe_type = fe_type
        if fe_type == 'resnet18': 
            dim = 512
            if pretrain: 
                self.fe.load_state_dict(
                    torch.load(
                        os.path.join(project_root, "Deps", "pretrained_fe", "resnet18-f37072fd.pth")
                    )
                )
        elif fe_type == 'resnet50':
            dim = 2048
            if pretrain: 
                self.fe.load_state_dict(
                    torch.load(
                        os.path.join(project_root, "Deps", "pretrained_fe", "resnet50-0676ba61.pth") # V1
                    )
                )
        elif fe_type == 'vit_b_16':
            dim = 768
            if pretrain: 
                self.fe.load_state_dict(
                    torch.load(
                        os.path.join(project_root, "Deps", "pretrained_fe", "vit_b_16-c867db91.pth") # V1
                    )
                )
        else:
            raise ValueError("No valid feature extractor!")
        
        self.ts_fc1 = nn.Linear(dim, 100, bias = False)
        self.ts_fc2 = nn.Linear(100, 100, bias = False)
        self.ts_fc3 = nn.Linear(100, 2, bias = False)
        self.ts_bn1 = nn.BatchNorm1d(100)
        self.ts_bn2 = nn.BatchNorm1d(100)

        self.ca_fc1 = nn.Linear(symbol_size, dim, bias = False)
        self.ca_fc2 = nn.Linear(dim, 100, bias = False)
        self.ca_fc3 = nn.Linear(100, 100, bias = False)
        self.ca_bn1 = nn.BatchNorm1d(dim)
        self.ca_bn2 = nn.BatchNorm1d(100)
        self.ca_bn3 = nn.BatchNorm1d(100) # after 20240406

        self.ts_afun = nn.ReLU()
        self.ca_afun = nn.Sigmoid()

        self.symbol_set = nn.Parameter(torch.rand((num_classes, symbol_size), requires_grad=True))

        self.t = nn.Parameter(torch.rand(1, requires_grad=True))
        self.new_symbol = nn.Parameter(torch.rand((num_new, symbol_size), requires_grad=True))
    
    def _fe(self, x):
        if 'resnet' in self.fe_type:
            x = self.fe.conv1(x)
            x = self.fe.bn1(x)
            x = self.fe.relu(x)
            x = self.fe.maxpool(x)

            x = self.fe.layer1(x)
            x = self.fe.layer2(x)
            x = self.fe.layer3(x)
            x = self.fe.layer4(x)

            x = self.fe.avgpool(x)
            x = torch.flatten(x, 1)
        elif 'vit' in self.fe_type:
            # Reshape and permute the input tensor
            x = self.fe._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.fe.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.fe.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

        return x

    def forward(self, x, symbol):
        if self.fix_fe: 
            self.fe.eval()
            with torch.no_grad(): x = self._fe(x)
        else: x = self._fe(x)

        symbol = self.ca_fc1(symbol)
        symbol = self.ca_bn1(symbol)
        symbol = self.ca_afun(symbol)

        x = torch.mul(x, symbol) # batch_size * 2048

        x = self.ts_fc1(x)
        x = self.ts_bn1(x)
        x = self.ts_afun(x)

        symbol = self.ca_fc2(symbol)
        symbol = self.ca_bn2(symbol)
        symbol = self.ca_afun(symbol)

        x = torch.mul(x, symbol) # batch_size * 100

        x = self.ts_fc2(x)
        x = self.ts_bn2(x)
        x = self.ts_afun(x)

        symbol = self.ca_fc3(symbol)
        symbol = self.ca_bn3(symbol) # after 20240406
        symbol = self.ca_afun(symbol)

        x = torch.mul(x, symbol) # batch_size * 100

        x = self.ts_fc3(x)

        return x
    
    def ts_forward(self, x):
        if self.fix_fe: 
            self.fe.eval()
            with torch.no_grad(): x = self._fe(x)
        else: x = self._fe(x)

        x = self.ts_fc1(x)
        x = self.ts_bn1(x)
        x = self.ts_afun(x)

        x = self.ts_fc2(x)
        x = self.ts_bn2(x)
        x = self.ts_afun(x)

        x = self.ts_fc3(x)

        return x

    def ts_feature_forward(self, x):
        x = self.ts_fc1(x)
        x = self.ts_bn1(x)
        x = self.ts_afun(x)

        x = self.ts_fc2(x)
        x = self.ts_bn2(x)
        x = self.ts_afun(x)

        x = self.ts_fc3(x)

        return x
    
    def feature_forward(self, x, symbol):
        symbol = self.ca_fc1(symbol)
        symbol = self.ca_bn1(symbol)
        symbol = self.ca_afun(symbol)

        x = torch.mul(x, symbol) # batch_size * 2048

        x = self.ts_fc1(x)
        x = self.ts_bn1(x)
        x = self.ts_afun(x)

        symbol = self.ca_fc2(symbol)
        symbol = self.ca_bn2(symbol)
        symbol = self.ca_afun(symbol)

        x = torch.mul(x, symbol) # batch_size * 100

        x = self.ts_fc2(x)
        x = self.ts_bn2(x)
        x = self.ts_afun(x)

        symbol = self.ca_fc3(symbol)
        symbol = self.ca_bn3(symbol) # after 20240406
        symbol = self.ca_afun(symbol)

        x = torch.mul(x, symbol) # batch_size * 100

        x = self.ts_fc3(x)

        return x

    def cls(self, x):
        XX = torch.cat([x for i in range(self.symbol_set.shape[0])], 0)
        with torch.no_grad():
            y_hat = self.forward(XX, self.symbol_set)
            y_hat = torch.softmax(y_hat, dim = 1)
        Y_pred = y_hat.argmax(dim = 0)
        
        return Y_pred[1]

    def symbol_orthg(self):
        device = self.symbol_set.device
        S_e = F.normalize(self.symbol_set, p = 2, dim = 1)
        logits = torch.mm(S_e, S_e.T) * torch.exp(self.t)
        
        logits = logits.to(device)
        labels = torch.arange(self.symbol_set.shape[0]).to(device)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, labels)

        return loss