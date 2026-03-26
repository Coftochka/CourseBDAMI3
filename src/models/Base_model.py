from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base — every model (torch and non-torch) implements this
# ──────────────────────────────────────────────────────────────────────────────

class BaseModel(ABC):

    @abstractmethod
    def fit(self, X, y, **kwargs): ...

    @abstractmethod
    def predict(self, X, **kwargs): ...

    @abstractmethod
    def save(self, path: str): ...


# ──────────────────────────────────────────────────────────────────────────────
# Shared base for all Torch sequence models (LSTM, GRU, CNN, Transformer)
# ──────────────────────────────────────────────────────────────────────────────

class TorchBaseModel(BaseModel, nn.Module):
    """
    Mixin for PyTorch regression models that operate on sliding windows.

    Subclass must define:
        self.seq_len  : int
        self.mode     : "single" | "pooled" | "finetune"
        forward(x, ticker_ids) -> Tensor

    Device is resolved automatically: CUDA > MPS > CPU.
    Pass device= explicitly to override.
    """

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def _make_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ticker_ids: Optional[np.ndarray] = None,
    ):
        """
        X          : (n_timesteps, n_features)
        y          : (n_timesteps,)
        ticker_ids : (n_timesteps,) int — optional

        Returns:
            X_win  : (n_windows, seq_len, n_features)
            y_win  : (n_windows,)
            ids_win: (n_windows,) int | None
        """
        X_win, y_win, ids_win = [], [], []
        max_i = len(X) - self.seq_len + 1
        for i in range(max_i):
            X_win.append(X[i: i + self.seq_len])
            y_win.append(y[i + self.seq_len - 1])
            if ticker_ids is not None:
                ids_win.append(ticker_ids[i + self.seq_len - 1])

        X_win = np.array(X_win, dtype=np.float32)
        y_win = np.array(y_win, dtype=np.float32)
        ids_win = np.array(ids_win, dtype=np.int64) if ids_win else None
        return X_win, y_win, ids_win

    def _build_loader(self, X_w, y_w, ids_w, batch_size: int, shuffle: bool):
        X_t = torch.tensor(X_w, dtype=torch.float32)
        y_t = torch.tensor(y_w, dtype=torch.float32)
        if ids_w is not None:
            ids_t = torch.tensor(ids_w, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_t, y_t, ids_t)
        else:
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader, dataset

    def _run_epoch(self, loader, optimizer, criterion, training: bool):
        self.train(training)
        device = next(self.parameters()).device
        total_loss = 0.0
        n = 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                if len(batch) == 3:
                    X_b, y_b, ids_b = [t.to(device) for t in batch]
                else:
                    X_b, y_b = [t.to(device) for t in batch]
                    ids_b = None

                if training:
                    optimizer.zero_grad()

                loss = criterion(self(X_b, ids_b), y_b)

                if training:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * len(X_b)
                n += len(X_b)
        return total_loss / n

    # ------------------------------------------------------------------
    # Shared fit / pretrain
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        ticker_ids=None,
        X_val=None,
        y_val=None,
        ticker_ids_val=None,
        optimizer=None,
        scheduler=None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        X, y         : training data. y — target (float), preprocessed.
        ticker_ids   : (n_timesteps,) int — required for pooled / finetune.

        mode="single":
            fit(X, y, X_val=..., y_val=...)

        mode="pooled":
            fit(X, y, ticker_ids=ids, X_val=..., y_val=..., ticker_ids_val=ids_val)

        mode="finetune":
            Use pretrain() and finetune() instead of fit() directly.
        """
        if self.mode != "single":
            assert ticker_ids is not None, f"ticker_ids is required for mode='{self.mode}'"

        X_w, y_w, ids_w = self._make_windows(X, y, ticker_ids)
        has_val = X_val is not None and y_val is not None

        val_loader = None
        if has_val:
            X_val_w, y_val_w, ids_val_w = self._make_windows(X_val, y_val, ticker_ids_val)
            val_loader, _ = self._build_loader(X_val_w, y_val_w, ids_val_w, batch_size, shuffle=False)

        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": []}

        train_loader, _ = self._build_loader(X_w, y_w, ids_w, batch_size, shuffle=True)

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer, criterion, training=True)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, optimizer, criterion, training=False)
                self.history["val_loss"].append(val_loss)

            if scheduler is not None:
                scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                val_str = f"  val_loss={val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.6f}{val_str}  lr={lr:.2e}")

    def pretrain(self, X, y, ticker_ids, X_val=None, y_val=None, ticker_ids_val=None, **fit_kwargs):
        """
        Stage 1: train on all assets at once.
        Only for mode="finetune".
        """
        assert self.mode == "finetune", "pretrain() is only available for mode='finetune'"
        print("=== Pretrain (all assets) ===")
        self.fit(
            X, y,
            ticker_ids=ticker_ids,
            X_val=X_val, y_val=y_val, ticker_ids_val=ticker_ids_val,
            **fit_kwargs,
        )

    # ------------------------------------------------------------------
    # Shared predict
    # ------------------------------------------------------------------

    def predict(self, X, ticker_ids=None) -> np.ndarray:
        """
        X          : (n_timesteps, n_features)
        ticker_ids : (n_timesteps,) int — required for pooled / finetune

        Returns:
            pred : (n_windows,) float
        """
        device = next(self.parameters()).device
        X_w, _, ids_w = self._make_windows(X, np.zeros(len(X)), ticker_ids)
        self.eval()
        X_t = torch.tensor(X_w, dtype=torch.float32).to(device)
        ids_t = torch.tensor(ids_w, dtype=torch.long).to(device) if ids_w is not None else None
        with torch.no_grad():
            pred = self(X_t, ids_t)
        return pred.cpu().numpy()

    def predict_last(self, X, ticker_id: Optional[int] = None) -> float:
        """
        X         : (n_timesteps, n_features) — at least seq_len rows
        ticker_id : int — required for pooled / finetune

        Returns:
            predicted return : float
        """
        if len(X) < self.seq_len:
            raise ValueError(f"need at least {self.seq_len} candles, got {len(X)}")

        device = next(self.parameters()).device
        window_t = torch.tensor(X[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(device)

        ids_t = None
        if self.mode != "single":
            assert ticker_id is not None, "ticker_id is required for pooled / finetune"
            ids_t = torch.tensor([ticker_id], dtype=torch.long).to(device)

        self.eval()
        with torch.no_grad():
            return self(window_t, ids_t).item()

    # ------------------------------------------------------------------
    # Shared metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _regression_metrics_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> pd.DataFrame:
        """
        y_true : real returns
        y_pred : predicted returns
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])

        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "dir_accuracy": dir_acc,
            "ic": ic,
        }
        return pd.DataFrame(metrics, index=[model_name])

    def scores(
        self,
        X,
        y,
        model_name: str,
        ticker_ids=None,
    ) -> pd.DataFrame:
        """
        Regression metrics on the sample.

        For multiple assets, pass ticker_ids — you'll get aggregated metrics.
        For metrics per asset separately, use scores_per_ticker().
        """
        _, y_w, _ = self._make_windows(X, y, ticker_ids)
        y_pred = self.predict(X, ticker_ids)
        return self._regression_metrics_df(y_w, y_pred, model_name)

    def scores_per_ticker(
        self,
        X_dict: Dict[str, np.ndarray],
        y_dict: Dict[str, np.ndarray],
        ticker_to_id: Dict[str, int],
    ) -> pd.DataFrame:
        """
        X_dict       : {"AAPL": X_aapl, "MSFT": X_msft, ...}
        y_dict       : {"AAPL": y_aapl, "MSFT": y_msft, ...}
        ticker_to_id : {"AAPL": 0, "MSFT": 1, ...} — required for pooled / finetune

        Returns:
            DataFrame — one row per ticker + "ALL" (aggregate) row
        """
        frames = []
        all_y, all_pred = [], []

        for ticker, X in X_dict.items():
            y = y_dict[ticker]
            ticker_ids = (
                None if self.mode == "single"
                else np.full(len(X), ticker_to_id[ticker], dtype=np.int64)
            )

            _, y_w, _ = self._make_windows(X, y, ticker_ids)
            y_pred = self.predict(X, ticker_ids)

            frames.append(self._regression_metrics_df(y_w, y_pred, ticker))
            all_y.append(y_w)
            all_pred.append(y_pred)

        frames.append(self._regression_metrics_df(
            np.concatenate(all_y),
            np.concatenate(all_pred),
            "ALL",
        ))
        return pd.concat(frames)

    # ------------------------------------------------------------------
    # Shared plot
    # ------------------------------------------------------------------

    def plot_loss(self):
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])
        if not train:
            raise RuntimeError("fit model before plotting loss")

        epochs = range(1, len(train) + 1)
        plt.figure(figsize=(9, 4))
        plt.plot(epochs, train, label="Train loss (MSE)")
        if val:
            plt.plot(epochs, val, label="Val loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Loss by epochs")
        plt.legend()
        plt.tight_layout()
        plt.show()
