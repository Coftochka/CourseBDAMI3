"""
Abstract base classes for windowed time-series regression models.

Interface contract (all models):
    model.fit(X_train, y_train, X_val=None, y_val=None) -> None
    model.predict(X) -> np.ndarray
    model.plot_loss(title="") -> None

Shapes:
    X      : (N, seq_len, num_features)   — 3-D tensor of windowed features
    y      : (N,)                          — scalar target per window
    predict: (M,)                          — one prediction per window
"""

from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import torch
import torch.utils.data
from torch import nn

# ──────────────────────────────────────────────────────────────────────────────
# Abstract base — every model (torch and non-torch) implements this
# ──────────────────────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """Minimal abstract interface shared by all regression models."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def plot_loss(self, title: str = "") -> None:
        """Plot training loss curve.  Override in subclasses that track loss."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Shared base for all PyTorch sequence models (LSTM, GRU, CNN, Transformer)
# ──────────────────────────────────────────────────────────────────────────────

class TorchBaseModel(BaseModel, nn.Module):
    """
    Common training / inference loop for PyTorch models.

    Subclass contract:
        1. Call ``nn.Module.__init__(self)`` in your ``__init__``.
        2. Set attributes: ``self.epochs``, ``self.batch_size``,
           ``self.lr``, ``self.patience``.
        3. Define ``forward(x: Tensor) -> Tensor``
           mapping (batch, seq_len, features) → (batch,).
        4. Call ``self.to(self._resolve_device(device))`` at the end of __init__.
    """

    # ── device helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, seq_len, features), got {X.ndim}-D")
        if y.shape != (X.shape[0],):
            raise ValueError(f"y shape {y.shape} != expected ({X.shape[0]},)")

        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_loader = self._make_loader(X, y, shuffle=True)

        has_val = X_val is not None and y_val is not None
        val_loader = None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.train_losses_: List[float] = []
        self.val_losses_: List[float] = []

        for _ in range(self.epochs):
            # — train —
            self.train()
            epoch_loss, epoch_n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
                epoch_n += len(xb)
            self.train_losses_.append(epoch_loss / epoch_n)

            # — validation & early stopping —
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, criterion, device)
                self.val_losses_.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(device)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D, got {X.ndim}-D")
        device = next(self.parameters()).device
        self.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size]).to(device)
                parts.append(self(xb).cpu().numpy())
        return np.concatenate(parts)

    # ── loss visualisation ─────────────────────────────────────────────────────

    def plot_loss(self, title: str = "") -> None:
        """Plot train (and optionally val) loss curves recorded during fit()."""
        import matplotlib.pyplot as plt

        if not getattr(self, "train_losses_", None):
            return
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.train_losses_, label="Train loss")
        if self.val_losses_:
            ax.plot(self.val_losses_, label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title or f"{self.__class__.__name__} training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y),
        )
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _eval_loss(self, loader, criterion, device) -> float:
        self.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(self(xb), yb)
                total += loss.item() * len(xb)
                n += len(xb)
        return total / n
