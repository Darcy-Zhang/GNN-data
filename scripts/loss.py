import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree

class weighted_loss(nn.Module):
    def __init__(self, edge_index, lamb):
        super().__init__()
        self.edge_index = edge_index
        self.lamb = lamb
        self.deg = degree(edge_index[0], 1000, dtype=torch.float)

    def forward(self, out, data):
        x_Ni_state = torch.zeros((1000, 2), device='cuda')
        state_dict = {}
        w = torch.zeros((1000), dtype=float, device='cuda')
        x = data.x
        y = data.y.view(-1).long()

        # row 节点i col 节点j
        for row, col in self.edge_index.t():
            # 如果节点j为I
            if x[col] > 0.999:
                x_Ni_state[row][0] += 1
            else:
                x_Ni_state[row][1] += 1
        
        # 统计每种情况的个数
        for i, state in enumerate(x_Ni_state):
            key = (x[i].item(), state[0].item(), state[1].item())
            if state_dict.get(key) != None:
                state_dict[key] += 1
            else:
                state_dict[key] = 1
        
        for i in range(len(w)):
            key = (x[i].item(), x_Ni_state[i][0].item(), x_Ni_state[i][1].item())
            w[i] = pow(state_dict[key], -self.lamb)
        
        # 正则化项
        Z = w.sum()

        node_loss = F.cross_entropy(out, y, reduction='none')


        loss = node_loss * w
        # print(loss)
        # print(loss.sum())
        return loss.sum() / Z

