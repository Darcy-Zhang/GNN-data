from message import Att_layer
import torch.nn as nn
from torch.nn import Linear, Sequential
import torch.nn.functional as F

class Att_Gnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_in = Sequential(
            Linear(1, 32),
            nn.ReLU(),
            Linear(32, 32),
            nn.ReLU()
        )

        self.att_layer = Att_layer(32)

        self.mlp_out = Sequential(
            Linear(32, 32),
            nn.ReLU(),
            Linear(32, 2)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.mlp_in(x)
        x = self.att_layer(x, edge_index)
        x = self.att_layer(x, edge_index)
        x = self.mlp_out(x)

        return F.softmax(x, dim=1)