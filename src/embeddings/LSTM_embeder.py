from torch import nn
from typing import Optional
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt


class LSTMEmbedder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)
        self.history: dict[str, list[float]] = {}

        dev = device or ("cuda" if torch.cuda.is_available() else
                         "mps"  if torch.backends.mps.is_available() else "cpu")
        self.to(dev)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode(x)).squeeze(-1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        clip_grad: float = 1.0,
        verbose: bool = True,
    ) -> "LSTMEmbedder":
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.history = {"train": [], "val": []}

        train_loader = self._loader(X, y, batch_size, shuffle=True)
        val_loader = self._loader(X_val, y_val, batch_size, shuffle=False) if X_val is not None else None

        for epoch in range(epochs):
            train_loss = self._epoch(train_loader, optimizer, criterion, training=True, clip_grad=clip_grad)
            self.history["train"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._epoch(val_loader, optimizer, criterion, training=False)
                self.history["val"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                val_str = f"  val={val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1:>3}/{epochs}  train={train_loss:.6f}{val_str}")

        return self

    def transform(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        device = next(self.parameters()).device
        self.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x_t = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
                chunks.append(self.encode(x_t).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def fit_transform(self, X, y, X_val=None, y_val=None, **fit_kwargs) -> np.ndarray:
        self.fit(X, y, X_val, y_val, **fit_kwargs)
        return self.transform(X)

    def plot_loss(self):
        plt.figure(figsize=(9, 4))
        plt.plot(self.history["train"], label="Train")
        if self.history.get("val"):
            plt.plot(self.history["val"], label="Val")
        plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout(); plt.show()

    def save(self, path: str):
        torch.save({
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "config": {"input_size": self.input_size, "hidden_size": self.hidden_size, "num_layers": self.num_layers},
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LSTMEmbedder":
        ckpt  = torch.load(path, weights_only=True, map_location="cpu")
        model = cls(**ckpt["config"], device=device)
        model.load_state_dict(ckpt["state_dict"])
        return model

    @staticmethod
    def _loader(X, y, batch_size, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _epoch(self, loader, optimizer, criterion, training: bool, clip_grad: float = 0.0) -> float:
        self.train(training)
        device = next(self.parameters()).device
        total, n = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                if training:
                    optimizer.zero_grad()
                loss = criterion(self(X_b), y_b)
                if training:
                    loss.backward()
                    if clip_grad > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                    optimizer.step()
                total += loss.item() * len(X_b)
                n += len(X_b)
        return total / n
