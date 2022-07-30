from model import Att_Gnn
from dataset import AttDataset
import torch

dataset = AttDataset(root='../data')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Att_Gnn().to(device)