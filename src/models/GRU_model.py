from model_base import BaseModel
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class GRUModel(BaseModel, nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int = 1, seq_len: int = 30):
        """
        input:
            input_size: number of features
            hidden_size: number of hidden units
            num_layers: number of GRU layers
            horizon: number of steps to predict (reserved for future use)
            seq_len: length of the input sequence window
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.seq_len = seq_len

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.history = {"train_loss": [], "val_loss": []}


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :]).squeeze(-1)


    def _make_windows(self, X: np.ndarray, y: np.ndarray):
        """
        input:
            X, y: ndarray (n_timesteps, n_features) / (n_timesteps,)
        output:
            X_win, y_win: ndarray (n_windows, seq_len, n_features) / (n_windows,)
        """
        X_win, y_win = [], []
        for i in range(len(X) - self.seq_len + 1):
            X_win.append(X[i : i + self.seq_len])
            y_win.append(y[i + self.seq_len - 1])
        return np.array(X_win, dtype=np.float32), np.array(y_win, dtype=np.float32)


    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, scheduler=None, epochs: int = 50, batch_size: int = 32, verbose: bool = True):
        """
        input:
            X, y: np.ndarray (n_timesteps, n_features) / (n_timesteps,)
            X_val, y_val : np.ndarray (n_timesteps, n_features) / (n_timesteps,)
            optimizer: torch optimizer (default: Adam lr=1e-3)
            scheduler: lr scheduler
            epochs: number of training epochs
            batch_size: mini-batch size
            verbose: print progress every 10 epochs
        """
        X_w, y_w = self._make_windows(X, y)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_w, y_val_w = self._make_windows(X_val, y_val)

        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        self.history = {"train_loss": [], "val_loss": []}

        X_t = torch.tensor(X_w, dtype=torch.float32)
        y_t = torch.tensor(y_w, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if has_val:
            X_val_t = torch.tensor(X_val_w, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_w, dtype=torch.float32)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self(X_b), y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(X_b)

            train_loss = total_loss / len(dataset)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if has_val:
                self.eval()
                with torch.no_grad():
                    val_loss = criterion(self(X_val_t), y_val_t).item()
                self.history["val_loss"].append(val_loss)

            if scheduler is not None:
                scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                val_str = f"  val_loss={val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.6f}{val_str}  lr={lr:.2e}")


    def plot_loss(self):
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])

        if not train:
            raise RuntimeError("fit model before plotting loss")

        epochs = range(1, len(train) + 1)
        plt.figure(figsize=(9, 4))
        plt.plot(epochs, train, label="Train loss")
        if val:
            plt.plot(epochs, val, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Loss by epochs")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict_proba(self, X) -> np.ndarray:
        """
        input:
            X: ndarray (n_timesteps, n_features)
        output:
            proba: ndarray (n_samples,) float ∈ [0, 1]
        """
        X_w, _ = self._make_windows(X, np.zeros(len(X)))
        self.eval()
        with torch.no_grad():
            proba = torch.sigmoid(self(torch.tensor(X_w, dtype=torch.float32)))
        return proba.numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """
        input:
            X: ndarray (n_timesteps, n_features)
            threshold: decision threshold
        output:
            pred: ndarray (n_samples,) bool
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_last(self, X, threshold: float = 0.5) -> tuple[float, bool]:
        """
        input:
            X: ndarray (n_timesteps, n_features)
            threshold: decision threshold
        output:
            proba: float ∈ [0, 1]
            pred: bool
        """
        if len(X) < self.seq_len:
            raise ValueError(f"need at least {self.seq_len} candles, got {len(X)}")

        window = X[-self.seq_len :]
        window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            proba = torch.sigmoid(self(window_t)).item()

        return proba, proba >= threshold

    def score(self, X, y) -> dict:
        """
        input:
            X: ndarray (n_timesteps, n_features)
            y: ndarray (n_timesteps,)
        output:
            accuracy: float
            f1: float
            roc_auc: float
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        _, y_w = self._make_windows(X, y)
        pred  = self.predict(X)
        proba = self.predict_proba(X)
        return {
            "accuracy": accuracy_score(y_w, pred),
            "f1":       f1_score(y_w, pred),
            "roc_auc":  roc_auc_score(y_w, proba),
        }


    def save(self, path: str):
        """
        input:
            path: str
        """
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_size":  self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers":  self.num_layers,
                "horizon":     self.horizon,
                "seq_len":     self.seq_len,
            }
        }, path)

    @classmethod
    def load(cls, path: str) -> "GRUModel":
        ckpt  = torch.load(path, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model