from .Base_model import TorchBaseModel
from torch import nn
from typing import Optional
import torch


class LSTMModel(TorchBaseModel):
    """LSTM regression model for windowed time-series."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.to(self._resolve_device(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → (batch,)"""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).squeeze(-1)
