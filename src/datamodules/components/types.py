from dataclasses import dataclass
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float) -> Tuple["InputData", "InputData"]:
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=self.y
        )
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))

    def dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            TensorDataset(self.x, self.y),
            **kwargs,
        )
