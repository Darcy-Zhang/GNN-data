import os
import os.path as osp
from typing import List

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip

from util import read_att_data


class AttDataset(InMemoryDataset):
    url = 'https://github.com/Darcy-Zhang/GNN-data/raw/main/data/BA-SIS.zip'

    def __init__(self,
                 root: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'Att-Data', 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'Att-Data', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ["BA-SIS.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = read_att_data(self.raw_dir, self.raw_file_names[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list is self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = AttDataset(root='../data')
    print(dataset[0])