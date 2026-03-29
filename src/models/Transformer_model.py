from Base_model import TorchBaseModel
from torch import nn
from typing import Optional, Dict, List
import torch
import numpy as np
import math


# Входные ряды X: (n_timesteps, n_features); для parquet SBER задайте
# input_size = schema.INPUT_SIZE и порядок признаков как в schema.FEATURE_COLS.
# forward принимает x: (batch, seq_len, input_size).
#
# Режимы: "single" | "pooled" | "finetune" (см. корневой Transformer_model).
# Таргет y строится снаружи; для окна X[i : i+seq_len] метка — y[i + seq_len - 1].


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(TorchBaseModel):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        seq_len: int = 30,
        mode: str = "single",
        num_tickers: int = 1,
        embedding_dim: int = 8,
        device: Optional[str] = None,
    ):
        """
        input_size        : число признаков на шаг (для SBER — schema.INPUT_SIZE)
        d_model           : dimension of embedding (must be divisible by nhead)
        nhead             : number of attention heads
        num_encoder_layers: number of TransformerEncoder layers
        dim_feedforward   : dimension of FFN
        dropout           : dropout
        seq_len           : length of input window
        mode              : "single" | "pooled" | "finetune"
        num_tickers       : number of assets (ignored when mode="single")
        embedding_dim     : dimension of asset embedding
        device            : "cuda" / "mps" / "cpu" (auto-detected if None)
        """
        assert mode in ("single", "pooled", "finetune"), (
            f"mode is {mode}, should be 'single', 'pooled' or 'finetune'"
        )

        nn.Module.__init__(self)
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.seq_len = seq_len
        self.mode = mode
        self.num_tickers = num_tickers
        self.embedding_dim = embedding_dim

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        if mode == "single":
            self.ticker_embedding = None
            self.fc = nn.Linear(d_model, 1)
        else:
            self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
            self.fc = nn.Linear(d_model + embedding_dim, 1)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.to(self._resolve_device(device))

    def forward(self, x: torch.Tensor, ticker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (batch, seq_len, input_size)
        ticker_ids : (batch,) int — only needed for pooled / finetune
        """
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        last = self.transformer_encoder(x)[:, -1, :]

        if self.mode == "single":
            return self.fc(last).squeeze(-1)

        assert ticker_ids is not None, "ticker_ids is required for pooled and finetune modes"
        assert self.ticker_embedding is not None
        emb = self.ticker_embedding(ticker_ids)
        return self.fc(torch.cat([last, emb], dim=1)).squeeze(-1)

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
        freeze_encoder: bool = True,
        **fit_kwargs,
    ):
        """
        Stage 2: fine-tune on a single asset.
        Only for mode="finetune", after pretrain().

        ticker_id      : index of the asset
        freeze_encoder : True — freezes encoder + projection, trains only fc + embedding
                         False — trains the entire model with smaller lr
        """
        assert self.mode == "finetune", "finetune() is only available for mode='finetune'"
        print(f"=== Finetune (ticker_id={ticker_id}, freeze_encoder={freeze_encoder}) ===")

        for param in list(self.input_projection.parameters()) + list(self.transformer_encoder.parameters()):
            param.requires_grad = not freeze_encoder

        ids = np.full(len(X), ticker_id, dtype=np.int64)
        ids_val = np.full(len(X_val), ticker_id, dtype=np.int64) if X_val is not None else None

        if freeze_encoder:
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

        for param in list(self.input_projection.parameters()) + list(self.transformer_encoder.parameters()):
            param.requires_grad = True

    def save(self, path: str):
        torch.save({
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "config": {
                "input_size":         self.input_size,
                "d_model":            self.d_model,
                "nhead":              self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "dim_feedforward":    self.dim_feedforward,
                "dropout":            self.dropout_rate,
                "seq_len":            self.seq_len,
                "mode":               self.mode,
                "num_tickers":        self.num_tickers,
                "embedding_dim":      self.embedding_dim,
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TransformerModel":
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        cfg = dict(ckpt["config"])
        cfg.pop("horizon", None)
        cfg["device"] = device
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
