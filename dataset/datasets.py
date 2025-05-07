# 定义各时序数据集成员类
from torch.utils.data import Dataset
import torch

class EnergyDataset(Dataset):
    def __init__(self, series_list, history_len, forward_len):
        self.X, self.y = [], []
        for series in series_list:
            for i in range(len(series) - history_len - forward_len + 1):
                self.X.append(series[i:i + history_len])
                self.y.append(series[i + history_len: i + history_len + forward_len])
        self.X = torch.tensor(self.X).unsqueeze(-1)
        self.y = torch.tensor(self.y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]