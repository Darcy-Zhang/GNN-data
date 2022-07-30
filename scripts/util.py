import pandas as pd
import os.path as osp
from io import StringIO

import torch
from torch_geometric.data import Data


def read_att_data(folder, prefix):

    raw_feature = []

    with open(osp.join(folder, prefix)) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "node," in line:
                N = int(line[5:])
            if "adjacency" in line:
                adj_line = i
                break

        # 取出N行数据
        adjlines = lines[adj_line + 1 : adj_line + N + 1]

        # 去除第一列
        for i, line in enumerate(adjlines):
            line = line.split(",", 1)[1:]
            adjlines[i] = ",".join(line)

        adjacency = pd.read_csv(StringIO("\n".join(adjlines)), header=None)

        # 获得稀疏矩阵
        edge_index = []
        for i, series in adjacency.iterrows():
            for j, value in series.items():
                if value == 1:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index).t().contiguous()

        for i, line in enumerate(lines):
            if "date" in line:
                temp_feature = (
                    torch.tensor(
                        pd.read_csv(StringIO("\n".join(lines[i : i + N + 1])))
                        .loc[:, ["I", "S"]]
                        .to_numpy()
                    )
                    .t()
                    .contiguous()
                )
                raw_feature.append(temp_feature)

    f.close()

    datalist = []

    for i in range(len(raw_feature) - 1):
        x = raw_feature[i][0].unsqueeze(1).contiguous().float()
        y = raw_feature[i + 1][1].unsqueeze(1).contiguous().float()
        datalist.append(Data(x=x, y=y, edge_index=edge_index))

    return datalist
