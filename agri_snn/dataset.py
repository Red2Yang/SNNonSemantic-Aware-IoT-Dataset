import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class AgriIoTDataset(Dataset):
    def __init__(self, X_ms, X_rgb, y_action):

        self.X_ms = torch.nan_to_num(torch.FloatTensor(X_ms), nan=0.0)
        self.X_rgb = torch.nan_to_num(torch.FloatTensor(X_rgb), nan=0.0)
        self.y_action = torch.LongTensor(y_action)

        self.ms_mask = torch.BoolTensor(~np.isnan(X_ms))
        self.rgb_mask = torch.BoolTensor(~np.isnan(X_rgb))

    def __len__(self):
        return len(self.X_ms)

    def __getitem__(self, idx):
        return (self.X_ms[idx], self.X_rgb[idx],
                self.ms_mask[idx], self.rgb_mask[idx],
                self.y_action[idx])


def create_dataloaders(X_ms, X_rgb, y_action,
                       test_ratio=0.2, batch_size=64, random_state=42):
    indices = np.arange(len(X_ms))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio,
        random_state=random_state, stratify=y_action
    )
    train_dataset = AgriIoTDataset(
        X_ms[train_idx], X_rgb[train_idx], y_action[train_idx]
    )
    test_dataset = AgriIoTDataset(
        X_ms[test_idx], X_rgb[test_idx], y_action[test_idx]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)
    print(f"[数据划分] 训练: {len(train_dataset)} 测试: {len(test_dataset)}")
    return train_loader, test_loader