from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover

    def _load_dotenv(*_a, **_k):
        pass

warnings.filterwarnings("ignore")


# ── split boundaries (календарь, как раньше) ───────────────────────────────────

@dataclass(frozen=True)
class SplitBounds:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


DEFAULT_SPLIT = SplitBounds(
    train_end=cast(pd.Timestamp, pd.Timestamp("2022-12-31")),
    val_start=cast(pd.Timestamp, pd.Timestamp("2023-01-01")),
    val_end=cast(pd.Timestamp, pd.Timestamp("2024-12-31")),
    test_start=cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
    test_end=cast(pd.Timestamp, pd.Timestamp("2026-03-06")),
)

TRAIN_END = DEFAULT_SPLIT.train_end
VAL_START = DEFAULT_SPLIT.val_start
VAL_END = DEFAULT_SPLIT.val_end
TEST_START = DEFAULT_SPLIT.test_start
TEST_END = DEFAULT_SPLIT.test_end


def _project_root() -> Path:
    # prep.py → data_analisys&prep → data → src → repo root
    return Path(__file__).resolve().parents[3]


def _default_candles_root() -> Path:
    _load_dotenv(_project_root() / ".env")
    return Path(os.getenv("DATA_ROOT", _project_root() / "src" / "data" / "moex_candles"))


class Dataloader:
    """
    Загрузка parquet-свечей, индикаторы, календарный train/val/test,
    нарезка окон с per-window нормализацией и таргетом log-return.

    Пример::

        loader = Dataloader()
        ds = loader.cut_on_windows(["SBER", "GAZP"])
        X_train, y_train, X_val, y_val, X_test, y_test = loader.concat_splits(ds)
    """

    def __init__(
        self,
        data_root: Path | str | None = None,
        split: SplitBounds | None = None,
        eps: float = 1e-8,
    ):
        self.data_root = Path(data_root) if data_root is not None else _default_candles_root()
        self.split = split or DEFAULT_SPLIT
        self.eps = eps

    # ── load / features / split ─────────────────────────────────────────────

    def load_df(
        self,
        ticker: str,
        interval: str = "daily",
        path: Path | str | None = None,
    ) -> pd.DataFrame | None:
        root = Path(path) if path is not None else self.data_root
        pth = root / interval / f"{ticker}.parquet"
        if not pth.exists():
            return None
        df = pd.read_parquet(pth)
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "begin", "time"):
                if col in df.columns:
                    df = df.set_index(pd.to_datetime(df[col])).drop(columns=[col])
                    break
        df.index.name = "timestamp"
        return df.sort_index()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()

        macd = ema12 - ema26
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        prev_close = c.shift(1)
        tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)

        df["sma5"] = c.rolling(5).mean()
        df["sma20"] = sma20
        df["ema12"] = ema12
        df["ema26"] = ema26
        df["close_sma20"] = c / sma20
        df["macd"] = macd
        df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd - df["macd_signal"]
        df["rsi14"] = 100 - 100 / (1 + gain / (loss + self.eps))
        df["bb_pct"] = (c - bb_lower) / (bb_upper - bb_lower + self.eps)
        df["bb_bw"] = (bb_upper - bb_lower) / (sma20 + self.eps)
        df["atr14"] = tr.ewm(span=14, adjust=False).mean()
        obv_raw = np.sign(delta.to_numpy(dtype=float, copy=False)) * v.to_numpy(
            dtype=float, copy=False
        )
        df["obv"] = pd.Series(obv_raw, index=df.index).fillna(0).cumsum()

        return df.dropna()

    def split_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        s = self.split
        train = df.loc[df.index <= s.train_end]
        val = df.loc[(df.index >= s.val_start) & (df.index <= s.val_end)]
        test = df.loc[df.index >= s.test_start]
        return train, val, test

    # ── windows ───────────────────────────────────────────────────────────────

    def make_windows(
        self,
        df: pd.DataFrame,
        seq_len: int = 30,
        step_size: int | None = None,
        horizon: int = 1,
        feature_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            X : (N, seq_len, n_features)
            y : (N,)  — log-return close_{t+horizon} / close_t
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns]
        step = step_size or seq_len

        values = df[feature_cols].values.astype(np.float32)
        close_all = df["close"].values.astype(np.float32)

        X, y = [], []
        for start in range(0, len(df) - seq_len - horizon + 1, step):
            end = start + seq_len

            window = values[start:end].copy()
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True)
            window_norm = (window - mean) / (std + self.eps)

            close_last = close_all[end - 1]
            close_next = close_all[end - 1 + horizon]
            target = (
                np.log(close_next / close_last)
                if close_last > 0 and close_next > 0
                else np.nan
            )

            X.append(window_norm)
            y.append(target)

        X_arr = np.array(X)
        y_arr = np.array(y, dtype=np.float32)
        mask = ~np.isnan(y_arr)
        return X_arr[mask], y_arr[mask]

    def cut_on_windows(
        self,
        tickers: list[str],
        interval: str = "daily",
        seq_len: int = 60,
        step_size: int | None = None,
        path: Path | str | None = None,
        horizon: int = 1,
        feature_cols: list[str] | None = None,
        add_indicators: bool = True,
    ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """
        Returns:
            dataset[ticker]["train|val|test"] = (X, y), X: (N, seq_len, F)
        """
        result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for ticker in tickers:
            df = self.load_df(ticker, interval, path)
            if df is None or len(df) < seq_len * 3:
                continue

            if add_indicators:
                df = self.add_features(df)
                if len(df) < seq_len * 3:
                    continue

            fcols = (
                [c for c in df.columns]
                if feature_cols is None
                else [c for c in feature_cols if c in df.columns]
            )
            if "close" not in fcols:
                fcols = ["close"] + fcols

            splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name, sdf in zip(("train", "val", "test"), self.split_df(df)):
                if len(sdf) < seq_len + horizon:
                    splits[name] = (np.empty((0, seq_len, len(fcols))), np.empty(0))
                else:
                    splits[name] = self.make_windows(
                        sdf, seq_len, step_size, horizon, fcols
                    )

            result[ticker] = splits

        return result

    @staticmethod
    def concat_splits(
        dataset: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Склеить все тикеры в один пул по train/val/test."""
        X_train = np.concatenate([dataset[t]["train"][0] for t in dataset])
        y_train = np.concatenate([dataset[t]["train"][1] for t in dataset])
        X_val = np.concatenate([dataset[t]["val"][0] for t in dataset])
        y_val = np.concatenate([dataset[t]["val"][1] for t in dataset])
        X_test = np.concatenate([dataset[t]["test"][0] for t in dataset])
        y_test = np.concatenate([dataset[t]["test"][1] for t in dataset])
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def splits_for_ticker(
        dataset: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
        ticker: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Один тикер: train/val/test массивы."""
        X_train = dataset[ticker]["train"][0]
        y_train = dataset[ticker]["train"][1]
        X_val = dataset[ticker]["val"][0]
        y_val = dataset[ticker]["val"][1]
        X_test = dataset[ticker]["test"][0]
        y_test = dataset[ticker]["test"][1]
        return X_train, y_train, X_val, y_val, X_test, y_test

