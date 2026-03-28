import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TRAIN_END  = pd.Timestamp("2022-12-31")
VAL_START  = pd.Timestamp("2023-01-01")
VAL_END    = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END   = pd.Timestamp("2026-03-06")


def load_df(ticker: str, interval: str = "daily", path: str = "data/moex_candles") -> pd.DataFrame | None:
    pth = Path(path) / interval / f"{ticker}.parquet"
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-8))
    df["bb_pct"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-8)
    df["bb_bw"] = (bb_upper - bb_lower) / (sma20 + 1e-8)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()
    df["obv"] = (np.sign(delta) * v).fillna(0).cumsum()

    return df.dropna()


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df.index <= TRAIN_END]
    val   = df.loc[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df.loc[df.index >= TEST_START]
    return train, val, test


def make_windows(
    df: pd.DataFrame,
    seq_len: int = 30,
    step_size: int | None = None,
    horizon: int = 1,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X : (N, seq_len, n_features)
        y : (N,)
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns]
    step = step_size or seq_len

    close_idx = feature_cols.index("close") if "close" in feature_cols else None
    values = df[feature_cols].values.astype(np.float32)

    close_all = df["close"].values.astype(np.float32)

    X, y = [], []
    for start in range(0, len(df) - seq_len - horizon + 1, step):
        end = start + seq_len

        window = values[start:end].copy()         
        mean = window.mean(axis=0, keepdims=True)  
        std  = window.std(axis=0, keepdims=True)  
        window_norm = (window - mean) / (std + 1e-8)

        close_last = close_all[end - 1]
        close_next = close_all[end - 1 + horizon]
        target = np.log(close_next / close_last) if close_last > 0 and close_next > 0 else np.nan

        X.append(window_norm)
        y.append(target)

    X_arr = np.array(X)
    y_arr = np.array(y, dtype=np.float32)
    mask  = ~np.isnan(y_arr)
    return X_arr[mask], y_arr[mask]


def cut_on_windows(
    tickers: list[str],
    interval: str = "daily",
    seq_len: int = 60,
    step_size: int | None = None,
    path: str = "data/moex_candles",
    horizon: int = 1,
    feature_cols: list[str] | None = None,
    add_indicators: bool = True,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Returns:
        dataset[ticker]["train"] = (X, y)   X: (N, seq_len, F)
        dataset[ticker]["val"]   = (X, y)
        dataset[ticker]["test"]  = (X, y)
    """
    result = {}
    for ticker in tickers:
        df = load_df(ticker, interval, path)
        if df is None or len(df) < seq_len * 3:
            continue

        if add_indicators:
            df = add_features(df)
            if len(df) < seq_len * 3:
                continue

        fcols = (
            [c for c in df.columns]
            if feature_cols is None
            else [c for c in feature_cols if c in df.columns]
        )
        if "close" not in fcols:
            fcols = ["close"] + fcols

        splits = {}
        for name, sdf in zip(("train", "val", "test"), split_df(df)):
            if len(sdf) < seq_len + horizon:
                splits[name] = (np.empty((0, seq_len, len(fcols))), np.empty(0))
            else:
                splits[name] = make_windows(sdf, seq_len, step_size, horizon, fcols)

        result[ticker] = splits

    return result


def get_train_val_test_data(dataset: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.concatenate([dataset[t]["train"][0] for t in dataset])
    y_train = np.concatenate([dataset[t]["train"][1] for t in dataset])
    X_val = np.concatenate([dataset[t]["val"][0] for t in dataset])
    y_val = np.concatenate([dataset[t]["val"][1] for t in dataset])
    X_test = np.concatenate([dataset[t]["test"][0] for t in dataset])
    y_test = np.concatenate([dataset[t]["test"][1] for t in dataset])
    return X_train, y_train, X_val, y_val, X_test, y_test




def get_train_val_test_data_by_ticker(dataset: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], ticker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = dataset[ticker]["train"][0]
    y_train = dataset[ticker]["train"][1]
    X_val = dataset[ticker]["val"][0]
    y_val = dataset[ticker]["val"][1]
    X_test = dataset[ticker]["test"][0]
    y_test = dataset[ticker]["test"][1]
    return X_train, y_train, X_val, y_val, X_test, y_test
