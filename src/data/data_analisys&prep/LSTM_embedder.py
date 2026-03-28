from torch import nn
from typing import Optional
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt


class LSTMEmbedder(nn.Module):
    """
    Supervised LSTM encoder.

    Принимает готовые окна (N, T, F), обучается предсказывать
    лог-доходность следующего бара, затем отдаёт скрытое состояние
    последнего шага как эмбеддинг окна.

        model = LSTMEmbedder(input_size=F, hidden_size=64)
        model.fit(X_train, y_train, X_val, y_val)
        emb = model.transform(X_test)   # (N, 64)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        input_size  : F — количество признаков на один бар
        hidden_size : размер скрытого состояния = размерность эмбеддинга
        num_layers  : число стекованных LSTM-слоёв
        dropout     : dropout между слоями (игнорируется при num_layers=1)
        device      : "cuda" / "cpu" / None (авто)
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        self.to(self._resolve_device(device))

    # ── device ────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ── forward / encode ──────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T, F)
        → (batch, hidden_size)  — скрытое состояние последнего шага
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction head (используется только при обучении)."""
        return self.fc(self.encode(x)).squeeze(-1)

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        epochs: int = 50,
        batch_size: int = 512,
        clip_grad: float = 1.0,
        verbose: bool = True,
    ) -> "LSTMEmbedder":
        """
        X        : (N, T, F)  float32 — готовые нормализованные окна
        y        : (N,)       float32 — таргет (лог-доходность)
        X_val    : (M, T, F)  — валидационные окна (опционально)
        y_val    : (M,)       — валидационный таргет
        clip_grad: порог gradient clipping (0 — выключить)
        """
        self._check_nans(X, "X")
        self._check_nans(y, "y")
        if X_val is not None:
            self._check_nans(X_val, "X_val")
        if y_val is not None:
            self._check_nans(y_val, "y_val")

        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": []}

        train_loader = self._make_loader(X, y, batch_size, shuffle=True)
        val_loader   = (
            self._make_loader(X_val, y_val, batch_size, shuffle=False)
            if X_val is not None and y_val is not None else None
        )

        for epoch in range(epochs):
            train_loss = self._epoch(train_loader, optimizer, criterion,
                                     training=True, clip_grad=clip_grad)
            self.history["train_loss"].append(train_loss)

            if np.isnan(train_loss):
                print(f"[!] train loss стал NaN на эпохе {epoch + 1}. "
                      "Проверьте данные и уменьшите lr.")
                break

            val_loss = None
            if val_loader is not None:
                val_loss = self._epoch(val_loader, optimizer, criterion,
                                       training=False, clip_grad=clip_grad)
                self.history["val_loss"].append(val_loss)

            if scheduler is not None:
                scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                lr      = optimizer.param_groups[0]["lr"]
                val_str = f"  val={val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1:>3}/{epochs}  train={train_loss:.6f}{val_str}  lr={lr:.2e}")

        return self

    # ── embedding extraction ──────────────────────────────────────────────────

    def transform(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """
        X       : (N, T, F)
        Returns : (N, hidden_size)
        """
        device = next(self.parameters()).device
        self.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x_t = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
                chunks.append(self.encode(x_t).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> torch.utils.data.DataLoader:
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _epoch(self, loader, optimizer, criterion, training: bool,
               clip_grad: float = 1.0) -> float:
        self.train(training)
        device     = next(self.parameters()).device
        total_loss = 0.0
        n          = 0
        ctx        = torch.enable_grad() if training else torch.no_grad()
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
                total_loss += loss.item() * len(X_b)
                n          += len(X_b)
        return total_loss / n

    @staticmethod
    def _check_nans(arr: np.ndarray, name: str):
        """Бросает ValueError с подробной диагностикой, если в массиве есть NaN/Inf."""
        nan_mask = ~np.isfinite(arr)
        n_bad    = int(nan_mask.sum())
        if n_bad == 0:
            return
        # координаты первого плохого значения
        first    = tuple(int(i) for i in np.argwhere(nan_mask)[0])
        raise ValueError(
            f"{name}: найдено {n_bad} NaN/Inf из {arr.size} элементов "
            f"(первый по индексу {first}). "
            "Проверьте препроцессинг: dropna, clip или fillna перед подачей в модель."
        )

    # ── diagnostics ───────────────────────────────────────────────────────────

    def plot_loss(self):
        train = self.history.get("train_loss", [])
        val   = self.history.get("val_loss",   [])
        if not train:
            raise RuntimeError("Сначала вызовите fit().")
        epochs = range(1, len(train) + 1)
        plt.figure(figsize=(9, 4))
        plt.plot(epochs, train, label="Train MSE")
        if val:
            plt.plot(epochs, val, label="Val MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("LSTMEmbedder — кривая обучения")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ── serialization ─────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
                "config": {
                    "input_size":  self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers":  self.num_layers,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LSTMEmbedder":
        ckpt  = torch.load(path, weights_only=True, map_location="cpu")
        cfg   = dict(ckpt["config"])
        cfg["device"] = device
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
