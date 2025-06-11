import torch
import torch.nn as nn

class Translator(nn.Module):
    def __init__(self, context_dim, num_hidden_layer, num_hidden_neuron, dropout_p):
        super(Translator, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(context_dim, num_hidden_neuron),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_p))
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layer):
            hidden_layer = nn.Linear(num_hidden_neuron, num_hidden_neuron)
            activation_layer = nn.ReLU()
            dropout_layer = nn.Dropout(dropout_p)
            self.hidden_layers.append(hidden_layer)
            self.hidden_layers.append(activation_layer)
            self.hidden_layers.append(dropout_layer)
        self.output_layer = nn.Linear(num_hidden_neuron, context_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.input_layer(x)
        for m in self.hidden_layers:
            x = m(x)
        output = self.output_layer(x)
        # output = self.tanh(output)

        return output


class TImodule(nn.Module):
    def __init__(self, context_dim, num_hidden_layer, num_hidden_neuron, dropout_p):
        super(TImodule, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(context_dim, num_hidden_neuron),
                                         nn.BatchNorm1d(num_hidden_neuron),                                   
                                         nn.Dropout(dropout_p),
                                         nn.ReLU())
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layer):
            hidden_layer = nn.Linear(num_hidden_neuron, num_hidden_neuron)
            bn_layer = nn.BatchNorm1d(num_hidden_neuron)
            activation_layer = nn.ReLU()
            dropout_layer = nn.Dropout(dropout_p)
            self.hidden_layers.append(hidden_layer)
            self.hidden_layers.append(bn_layer)
            self.hidden_layers.append(dropout_layer)
            self.hidden_layers.append(activation_layer)
        self.output_layer = nn.Linear(num_hidden_neuron, context_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.input_layer(x)
        for m in self.hidden_layers:
            x = m(x)
        output = self.output_layer(x)

        return output