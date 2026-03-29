"""
Базовые классы для регрессии по окнам свечей (пакет sber).

Пайплайн данных (вне моделей):
  - Загрузка parquet, сортировка по TIME_COLUMN (schema.TIME_COLUMN).
  - Таргет y (например лог-доходность следующей свечи) — тот же индекс, что у строк X.
  - X: только FEATURE_COLS из schema. Данные в модель подаются уже оконными
    (как после prepare_windows в ноутбуке): fit/predict/scores — одна и та же форма.
  - X: (n_samples, seq_len, n_features), y: (n_samples,). В forward батч: (batch, seq_len, n_features).

См. также: schema.FEATURE_COLS, schema.INPUT_SIZE, get_sber_feature_spec().
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_sber_feature_spec() -> Tuple[List[str], int, str]:
    """Имена признаков в порядке столбцов X, их число, имя колонки времени."""
    from .schema import FEATURE_COLS, INPUT_SIZE, TIME_COLUMN

    return list(FEATURE_COLS), INPUT_SIZE, TIME_COLUMN


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
    PyTorch-модели по батчу готовых окон (нарезка только снаружи, например prepare_windows).

    fit / predict / scores:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,) — только для fit и scores
        ticker_ids: (n_samples,) int — для pooled / finetune; для single не передаётся

    predict_last:
        X: одно окно (seq_len, n_features) или (1, seq_len, n_features)

    Subclass must define:
        self.seq_len  : int
        self.mode     : "single" | "pooled" | "finetune"
        forward(x, ticker_ids) -> Tensor

    Device: CUDA > MPS > CPU, либо device= в подклассе.
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
    # Окна: только проверка формы (нарезка — в prepare_windows и т.п.)
    # ------------------------------------------------------------------

    def _as_windowed_batch(self, X, ticker_ids):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n_samples, seq_len, n_features), got shape {X.shape}"
            )
        if X.shape[1] != self.seq_len:
            raise ValueError(
                f"X: expected seq_len={self.seq_len}, got {X.shape[1]}"
            )
        if hasattr(self, "input_size") and X.shape[2] != self.input_size:
            raise ValueError(
                f"X: expected n_features={self.input_size}, got {X.shape[2]}"
            )
        n = X.shape[0]
        if self.mode != "single":
            if ticker_ids is None:
                raise ValueError(f"ticker_ids is required for mode='{self.mode}'")
            ids = np.asarray(ticker_ids, dtype=np.int64)
            if ids.shape != (n,):
                raise ValueError(
                    f"ticker_ids must be shape ({n},), got {ids.shape}"
                )
            return X, ids
        return X, None

    @staticmethod
    def _align_y(y, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (n_samples,):
            raise ValueError(
                f"y must be shape ({n_samples},), got {y.shape}"
            )
        return y

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
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_prefix: str = "weights",
        save_optimizer_state: bool = False,
    ):
        """
        X, y         : оконные данные — X (n_samples, seq_len, n_features), y (n_samples,).
        ticker_ids   : (n_samples,) int — для pooled / finetune.

        mode="single":
            fit(X, y, X_val=..., y_val=...)

        mode="pooled":
            fit(X, y, ticker_ids=ids, X_val=..., y_val=..., ticker_ids_val=ids_val)

        mode="finetune":
            Use pretrain() and finetune() instead of fit() directly.

        checkpoint_dir:
            Если задано, после каждой эпохи сохраняется файл
            ``{checkpoint_prefix}_epoch_{epoch:04d}.pt`` с ключом ``state_dict`` (веса на CPU).
            При ``save_optimizer_state=True`` добавляется ``optimizer_state_dict`` для возобновления обучения.
        """
        X_w, ids_w = self._as_windowed_batch(X, ticker_ids)
        y_w = self._align_y(y, X_w.shape[0])
        has_val = X_val is not None and y_val is not None

        val_loader = None
        if has_val:
            X_val_w, ids_val_w = self._as_windowed_batch(X_val, ticker_ids_val)
            y_val_w = self._align_y(y_val, X_val_w.shape[0])
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

            if checkpoint_dir is not None:
                ckpt_root = Path(checkpoint_dir)
                ckpt_root.mkdir(parents=True, exist_ok=True)
                fname = f"{checkpoint_prefix}_epoch_{epoch + 1:04d}.pt"
                payload = {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                if save_optimizer_state:
                    payload["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(payload, ckpt_root / fname)

            if verbose and (epoch + 1) % 3 == 0:
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
    '''
        def predict(self, X, ticker_ids=None) -> np.ndarray:
            """
            X          : (n_samples, seq_len, n_features) — как в fit
            ticker_ids : (n_samples,) int — для pooled / finetune

            Returns:
                pred : (n_samples,) float
            """
            device = next(self.parameters()).device
            X_w, ids_w = self._as_windowed_batch(X, ticker_ids)
            self.eval()
            X_t = torch.tensor(X_w, dtype=torch.float32).to(device)
            ids_t = torch.tensor(ids_w, dtype=torch.long).to(device) if ids_w is not None else None
            with torch.no_grad():
                pred = self(X_t, ids_t)
            return pred.cpu().numpy()
    '''

    def predict(self, X, ticker_ids=None, batch_size: int = 256) -> np.ndarray:
        """
        X          : (n_samples, seq_len, n_features) — как в fit
        ticker_ids : (n_samples,) int — для pooled / finetune
        batch_size : int — размер батча для инференса

        Returns:
            pred : (n_samples,) float
        """
        # Используем существующий метод для проверки формата
        X_w, ids_w = self._as_windowed_batch(X, ticker_ids)
        
        device = next(self.parameters()).device
        self.eval()
        
        # Создаем фиктивные y для _build_loader (они не используются)
        dummy_y = np.zeros(len(X_w), dtype=np.float32)
        
        # Используем существующий _build_loader для батчированной инференса
        loader, _ = self._build_loader(X_w, dummy_y, ids_w, batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    X_b, _, ids_b = [t.to(device) for t in batch]
                else:
                    X_b, _ = [t.to(device) for t in batch]
                    ids_b = None
                
                pred_b = self(X_b, ids_b)
                all_preds.append(pred_b.cpu().numpy())
        
        return np.concatenate(all_preds)

    def predict_last(self, X, ticker_id: Optional[int] = None) -> float:
        """
        X         : (seq_len, n_features), (1, seq_len, n_features) или (N, seq_len, n_features)
                    — в последнем случае используется последнее окно X[-1]
        ticker_id : int — для pooled / finetune (один id на окно)

        Returns:
            predicted return : float
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            # одно окно (1, L, F) или весь тестовый батч (N, L, F) — берём последнее окно
            X = X[-1]
        if X.ndim != 2:
            raise ValueError(f"predict_last: ожидается (seq_len, n_features), got {X.shape}")
        if X.shape[0] != self.seq_len:
            raise ValueError(f"predict_last: нужно seq_len={self.seq_len}, got {X.shape[0]}")
        if hasattr(self, "input_size") and X.shape[1] != self.input_size:
            raise ValueError(
                f"predict_last: нужно n_features={self.input_size}, got {X.shape[1]}"
            )

        device = next(self.parameters()).device
        window_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

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
        batch_size=256
    ) -> pd.DataFrame:
        """
        Regression metrics on the sample.

        For multiple assets, pass ticker_ids — you'll get aggregated metrics.
        For metrics per asset separately, use scores_per_ticker().
        """
        X_w, _ = self._as_windowed_batch(X, ticker_ids)
        y_w = self._align_y(y, X_w.shape[0])
        y_pred = self.predict(X_w, ticker_ids, batch_size=batch_size)
        return self._regression_metrics_df(y_w, y_pred, model_name)

    def scores_per_ticker(
        self,
        X_dict: Dict[str, np.ndarray],
        y_dict: Dict[str, np.ndarray],
        ticker_to_id: Dict[str, int],
    ) -> pd.DataFrame:
        """
        X_dict       : тикер → X (n_samples, seq_len, n_features)
        y_dict       : тикер → y (n_samples,)
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
                else np.full(X.shape[0], ticker_to_id[ticker], dtype=np.int64)
            )

            X_w, _ = self._as_windowed_batch(X, ticker_ids)
            y_w = self._align_y(y, X_w.shape[0])
            y_pred = self.predict(X_w, ticker_ids)

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
