from Base_model import TorchBaseModel
from torch import nn
from typing import Optional, Dict, List
import torch
import numpy as np


# X: (n_timesteps, n_features); для SBER — schema.FEATURE_COLS / schema.INPUT_SIZE.
# Режимы и таргет — как в корневом LSTM_model.


class LSTMModel(TorchBaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_len: int = 30,
        mode: str = "single",
        num_tickers: int = 1,
        embedding_dim: int = 8,
        device: Optional[str] = None,
    ):
        """
        input_size    : number of features (для SBER — schema.INPUT_SIZE)
        hidden_size   : size of the hidden layer LSTM
        num_layers    : number of LSTM layers
        seq_len       : length of the input window
        mode          : "single" | "pooled" | "finetune"
        num_tickers   : number of assets (ignored for mode="single")
        embedding_dim : dimension of the asset embedding
        device        : "cuda" / "mps" / "cpu" (auto-detected if None)
        """
        assert mode in ("single", "pooled", "finetune"), (
            f"mode is {mode}, should be 'single', 'pooled' or 'finetune'"
        )

        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.mode = mode
        self.num_tickers = num_tickers
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        if mode == "single":
            self.ticker_embedding = None
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
            self.fc = nn.Linear(hidden_size + embedding_dim, 1)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.to(self._resolve_device(device))

    def forward(self, x: torch.Tensor, ticker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (batch, seq_len, input_size)
        ticker_ids : (batch,) int — only needed for pooled / finetune
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        lstm_out = out[:, -1, :]

        if self.mode == "single":
            return self.fc(lstm_out).squeeze(-1)

        assert ticker_ids is not None, "ticker_ids is required for pooled and finetune modes"
        assert self.ticker_embedding is not None
        emb = self.ticker_embedding(ticker_ids)
        return self.fc(torch.cat([lstm_out, emb], dim=1)).squeeze(-1)

    def finetune(
        self,
        X,
        y,
        ticker_id: int,
        X_val=None,
        y_val=None,
        optimizer=None,
        scheduler=None,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: bool = True,
        freeze_lstm: bool = True,
        **fit_kwargs,
    ):
        """
        Stage 2: fine-tune on a single asset.
        Only for mode="finetune", after pretrain().

        ticker_id   : index of the asset
        freeze_lstm : True — freezes LSTM, trains only fc + embedding
                      False — trains the entire model with smaller lr
        """
        assert self.mode == "finetune", "finetune() is only available for mode='finetune'"
        print(f"=== Finetune (ticker_id={ticker_id}, freeze_lstm={freeze_lstm}) ===")

        for param in self.lstm.parameters():
            param.requires_grad = not freeze_lstm

        ids = np.full(len(X), ticker_id, dtype=np.int64)
        ids_val = np.full(len(X_val), ticker_id, dtype=np.int64) if X_val is not None else None

        if freeze_lstm:
            assert self.ticker_embedding is not None
            trainable = list(self.fc.parameters()) + list(self.ticker_embedding.parameters())
            optimizer = optimizer or torch.optim.Adam(trainable, lr=1e-4)
        else:
            optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-4)

        self.fit(
            X, y,
            ticker_ids=ids,
            X_val=X_val, y_val=y_val, ticker_ids_val=ids_val,
            optimizer=optimizer, scheduler=scheduler,
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            **fit_kwargs,
        )

        for param in self.lstm.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save({
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "config": {
                "input_size":    self.input_size,
                "hidden_size":   self.hidden_size,
                "num_layers":    self.num_layers,
                "seq_len":       self.seq_len,
                "mode":          self.mode,
                "num_tickers":   self.num_tickers,
                "embedding_dim": self.embedding_dim,
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LSTMModel":
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        cfg = dict(ckpt["config"])
        cfg.pop("horizon", None)
        cfg["device"] = device
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
