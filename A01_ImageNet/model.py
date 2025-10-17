from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from pathlib import Path
import os

script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parent.parent

class cats_net(nn.Module):
    def __init__(
        self, 
        symbol_size, 
        num_new = 12, 
        num_classes = 10, 
        mlp_layers = 3,
        hidden_dim = 100,
        fix_fe = True, 
        fe_type = 'resnet18', 
        pretrain = False
    ):
        super().__init__()
        self.fix_fe = fix_fe
        self.fe = torchvision.models.__dict__[fe_type](weights=None)
        self.fe_type = fe_type
        self.mlp_layers = mlp_layers
        self.hidden_dim = hidden_dim
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
        
        for i in range(1, mlp_layers + 1):
            if i == 1 and mlp_layers == 1:
                # Single layer case: from feature dimension directly to output dimension
                setattr(self, f'ts_fc{i}', nn.Linear(dim, 2, bias=False))
                # Single layer does not need BatchNorm
            elif i == 1:
                # First layer in multi-layer case: from feature dimension to hidden dimension
                setattr(self, f'ts_fc{i}', nn.Linear(dim, hidden_dim, bias=False))
                setattr(self, f'ts_bn{i}', nn.BatchNorm1d(hidden_dim))
            elif i == mlp_layers:
                # Last layer in multi-layer case: from hidden dimension to output dimension
                setattr(self, f'ts_fc{i}', nn.Linear(hidden_dim, 2, bias=False))
                # Last layer does not need BatchNorm
            else:
                # Middle layers: hidden dimension to hidden dimension
                setattr(self, f'ts_fc{i}', nn.Linear(hidden_dim, hidden_dim, bias=False))
                setattr(self, f'ts_bn{i}', nn.BatchNorm1d(hidden_dim))

        # Dynamically create ca layers
        for i in range(1, mlp_layers + 1):
            if i == 1 and mlp_layers == 1:
                # Single layer case: from symbol_size to feature dimension (match ts layer input)
                setattr(self, f'ca_fc{i}', nn.Linear(symbol_size, dim, bias=False))
                setattr(self, f'ca_bn{i}', nn.BatchNorm1d(dim))
            elif i == 1:
                # First layer in multi-layer case: from symbol_size to feature dimension
                setattr(self, f'ca_fc{i}', nn.Linear(symbol_size, dim, bias=False))
                setattr(self, f'ca_bn{i}', nn.BatchNorm1d(dim))
            else:
                # All subsequent layers: from previous output dimension to current ts layer input dimension
                if i == 2:
                    # Second layer: from feature dimension to hidden dimension
                    setattr(self, f'ca_fc{i}', nn.Linear(dim, hidden_dim, bias=False))
                else:
                    # Third and later layers: from hidden dimension to hidden dimension
                    setattr(self, f'ca_fc{i}', nn.Linear(hidden_dim, hidden_dim, bias=False))
                setattr(self, f'ca_bn{i}', nn.BatchNorm1d(hidden_dim))

        self.ts_afun = nn.ReLU()
        self.ca_afun = nn.Sigmoid()

        # Initialize symbol_set with default random values
        self.symbol_set = nn.Parameter(torch.rand((num_classes, symbol_size), requires_grad=True))

        self.t = nn.Parameter(torch.rand(1, requires_grad=True))
        self.new_symbol = nn.Parameter(torch.rand((num_new, symbol_size), requires_grad=True))
    
    def init_symbol_set(self, init_type='random', custom_path=None):
        """Initialize symbol_set with different methods
        
        Args:
            init_type (str): 'random', 'one_hot', or 'word2vec'
            custom_path (str): path to word2vec symbol_set file (for 'word2vec' type)
        """
        num_classes, symbol_size = self.symbol_set.shape
        
        if init_type == 'random':
            # Already initialized with random values, no need to change
            pass
        elif init_type == 'one_hot':
            # Initialize with one-hot vectors
            if symbol_size < num_classes:
                raise ValueError(f"Symbol size ({symbol_size}) must be >= num_classes ({num_classes}) for one-hot initialization")
            
            one_hot_matrix = torch.zeros((num_classes, symbol_size))
            for i in range(num_classes):
                one_hot_matrix[i, i] = 1.0
            
            with torch.no_grad():
                self.symbol_set.data = one_hot_matrix
        elif init_type == 'word2vec':
            if custom_path is None:
                raise ValueError("custom_path must be provided for custom initialization")
            
            # Load custom symbol_set from file
            try:
                custom_symbols = torch.load(custom_path)
                if custom_symbols.shape != (num_classes, symbol_size):
                    raise ValueError(f"Custom symbol shape {custom_symbols.shape} doesn't match expected shape {(num_classes, symbol_size)}")
                
                with torch.no_grad():
                    self.symbol_set.data = custom_symbols
            except Exception as e:
                raise ValueError(f"Failed to load custom symbol_set from {custom_path}: {e}")
        else:
            raise ValueError(f"Unknown init_type: {init_type}. Must be 'random', 'one_hot', or 'word2vec'")
    
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

        return self._process_ca_ts_layers(x, symbol)
    
    def ts_forward(self, x):
        if self.fix_fe: 
            self.fe.eval()
            with torch.no_grad(): x = self._fe(x)
        else: x = self._fe(x)

        return self._process_ts_layers(x)

    def ts_feature_forward(self, x):
        return self._process_ts_layers(x)
    
    def feature_forward(self, x, symbol):
        return self._process_ca_ts_layers(x, symbol)

    def _process_ts_layers(self, x):
        """Process input through ts layers with dynamic layer count"""
        for i in range(1, self.mlp_layers + 1):
            ts_fc = getattr(self, f'ts_fc{i}')
            x = ts_fc(x)
            
            # Last layer does not need BatchNorm and activation function
            if i < self.mlp_layers:
                ts_bn = getattr(self, f'ts_bn{i}')
                x = ts_bn(x)
                x = self.ts_afun(x)
        return x

    def _process_ca_ts_layers(self, x, symbol):
        """Process input through ca and ts layers with dynamic layer count"""
        for i in range(1, self.mlp_layers + 1):
            # Process ca layer
            ca_fc = getattr(self, f'ca_fc{i}')
            ca_bn = getattr(self, f'ca_bn{i}')
            symbol = ca_fc(symbol)
            symbol = ca_bn(symbol)
            symbol = self.ca_afun(symbol)
            
            # element-wise multiplication
            x = torch.mul(x, symbol)
            
            # Process ts layer
            ts_fc = getattr(self, f'ts_fc{i}')
            x = ts_fc(x)
            
            # Last layer does not need BatchNorm and activation function
            if i < self.mlp_layers:
                ts_bn = getattr(self, f'ts_bn{i}')
                x = ts_bn(x)
                x = self.ts_afun(x)
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

class ts_net(nn.Module):
    def __init__(
        self, 
        mlp_layers = 3,
        hidden_dim = 100,
        fix_fe = True, 
        fe_type = 'resnet18', 
        pretrain = False
    ):
        super().__init__()
        self.fix_fe = fix_fe
        self.fe = torchvision.models.__dict__[fe_type](weights=None)
        self.fe_type = fe_type
        self.mlp_layers = mlp_layers
        self.hidden_dim = hidden_dim
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
        
        for i in range(1, mlp_layers + 1):
            if i == 1 and mlp_layers == 1:
                # Single layer case: from feature dimension directly to output dimension
                setattr(self, f'ts_fc{i}', nn.Linear(dim, 2, bias=False))
                # Single layer does not need BatchNorm
            elif i == 1:
                # First layer in multi-layer case: from feature dimension to hidden dimension
                setattr(self, f'ts_fc{i}', nn.Linear(dim, hidden_dim, bias=False))
                setattr(self, f'ts_bn{i}', nn.BatchNorm1d(hidden_dim))
            elif i == mlp_layers:
                # Last layer in multi-layer case: from hidden dimension to output dimension
                setattr(self, f'ts_fc{i}', nn.Linear(hidden_dim, 2, bias=False))
                # Last layer does not need BatchNorm
            else:
                # Middle layers: hidden dimension to hidden dimension
                setattr(self, f'ts_fc{i}', nn.Linear(hidden_dim, hidden_dim, bias=False))
                setattr(self, f'ts_bn{i}', nn.BatchNorm1d(hidden_dim))

        self.ts_afun = nn.ReLU()
    
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

    def forward(self, x):
        if self.fix_fe: 
            self.fe.eval()
            with torch.no_grad(): x = self._fe(x)
        else: x = self._fe(x)

        return self._process_ts_layers(x)

    def feature_forward(self, x):
        return self._process_ts_layers(x)

    def _process_ts_layers(self, x):
        """Process input through ts layers with dynamic layer count"""
        for i in range(1, self.mlp_layers + 1):
            ts_fc = getattr(self, f'ts_fc{i}')
            x = ts_fc(x)
            
            # Last layer does not need BatchNorm and activation function
            if i < self.mlp_layers:
                ts_bn = getattr(self, f'ts_bn{i}')
                x = ts_bn(x)
                x = self.ts_afun(x)
        return x
