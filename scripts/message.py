from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

import torch
import torch.nn.functional as F

class Att_layer(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr='add')
        self.A = Linear(in_channels, 1, bias=True)
        self.B = Linear(in_channels, 1, bias=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.A.reset_parameters()
        self.B.reset_parameters()
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self,x_i: Tensor , x_j: Tensor) -> Tensor:
        alpha_source = self.A(x_i)
        alpha_target = self.B(x_j)
        alpha = torch.sigmoid(alpha_source + alpha_target)
        return alpha * x_j

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return aggr_out + x
        