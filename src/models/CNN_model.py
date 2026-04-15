from .Base_model import TorchBaseModel
from torch import nn
from typing import Optional
import torch


class CNNModel(TorchBaseModel):

    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.num_filters = num_filters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        blocks = []
        in_ch = input_size
        for _ in range(num_layers):
            blocks += [
                nn.Conv1d(in_ch, num_filters, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = num_filters
        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, 1)
        self.to(self._resolve_device(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)               
        x = self.conv_blocks(x)               
        x = self.pool(x).squeeze(-1)          
        return self.fc(x).squeeze(-1)         
