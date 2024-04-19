import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ConvNet(nn.Module):

    def __init__(self, layer_dims, dropout_rate=0.0, device='cpu'):
        super(ConvNet, self).__init__()
        self.layer_dims = layer_dims
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim = 1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()
        
        self.input_layer = nn.Conv2d(in_channels=layer_dims[0], out_channels=layer_dims[1], 
                               kernel_size=4, stride=2, padding=1)
        
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(layer_dims) - 2):
            self.hidden_layers.append(nn.Conv2d(in_channels=layer_dims[i], out_channels=layer_dims[i+1], 
                               kernel_size=4, stride=2, padding=1))
        
        self.output_layer = nn.Linear(2*2*layer_dims[-2], layer_dims[-1])
        
        self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
            x = self.dropout(x)
            x = self.pool(x)

        # pdb.set_trace()
        x = self.flat(x)
        
        x = self.output_layer(x)
        x = self.softmax(x)
        return x