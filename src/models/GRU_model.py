from Base_model import TorchBaseModel
from torch import nn
from typing import Optional, Dict, List
import torch
import numpy as np


# Режимы работы модели:
#
#   "single"   — одна акция, ticker_ids не нужны.
#
#   "pooled"   — все акции сразу, одна общая модель.
#                Каждой акции присваивается embedding-вектор.
#                ticker_ids передаются в fit/predict.
#
#   "finetune" — двухэтапное обучение:
#                1. pretrain(X, y, ticker_ids) — обучение на всех акциях
#                2. finetune(X, y, ticker_id)  — дообучение на одной акции
#                   (замораживает GRU, обучает только fc + embedding)
#
# Таргет y строим заранее (shift, лог-доходность).
# Для окна X[i : i+seq_len] метка — y[i + seq_len - 1] (последний бар окна).


class GRUModel(TorchBaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_len: int = 30,
        mode: str = "single",
        num_tickers: int = 1,
        embedding_dim: int = 8,
    ):
        """
        input_size    : number of features
        hidden_size   : size of the hidden layer GRU
        num_layers    : number of GRU layers
        seq_len       : length of the input window
        mode          : "single" | "pooled" | "finetune"
        num_tickers   : number of assets (ignored for mode="single")
        embedding_dim : dimension of the asset embedding
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

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        if mode == "single":
            self.ticker_embedding = None
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
            self.fc = nn.Linear(hidden_size + embedding_dim, 1)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def forward(self, x: torch.Tensor, ticker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (batch, seq_len, input_size)
        ticker_ids : (batch,) int — only needed for pooled / finetune
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        gru_out = out[:, -1, :]

        if self.mode == "single":
            return self.fc(gru_out).squeeze(-1)

        assert ticker_ids is not None, "ticker_ids is required for pooled and finetune modes"
        emb = self.ticker_embedding(ticker_ids)
        return self.fc(torch.cat([gru_out, emb], dim=1)).squeeze(-1)

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
        freeze_gru: bool = True,
    ):
        """
        Stage 2: fine-tune on a single asset.
        Only for mode="finetune", after pretrain().

        ticker_id  : index of the asset
        freeze_gru : True — freezes GRU, trains only fc + embedding
                     False — trains the entire model with smaller lr
        """
        assert self.mode == "finetune", "finetune() is only available for mode='finetune'"
        print(f"=== Finetune (ticker_id={ticker_id}, freeze_gru={freeze_gru}) ===")

        for param in self.gru.parameters():
            param.requires_grad = not freeze_gru

        ids = np.full(len(X), ticker_id, dtype=np.int64)
        ids_val = np.full(len(X_val), ticker_id, dtype=np.int64) if X_val is not None else None

        if freeze_gru:
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
        )

        for param in self.gru.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
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
    def load(cls, path: str) -> "GRUModel":
        ckpt = torch.load(path, weights_only=True)
        cfg = dict(ckpt["config"])
        cfg.pop("horizon", None)
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
