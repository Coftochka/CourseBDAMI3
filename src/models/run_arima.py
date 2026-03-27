"""
Run ArimaModel on three hourly-candle tickers (SBER, GAZP, LKOH).

- target y  : log-return of close price
- split     : 70 / 15 / 15
- auto_order: True  → pmdarima.auto_arima selects (p, d, q)
- saves each fitted model to  ../../arima_saved/<TICKER>_arima.pkl
- prints regression metrics for every ticker
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from Arima_model import ArimaModel

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "moex_candles_hour")
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "arima_saved")
TICKERS    = ["SBER", "GAZP", "LKOH"]
N_ROWS     = 2000   # last N hourly bars — keeps rolling predict fast enough


def load_log_returns(ticker: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df   = pd.read_csv(path, parse_dates=["date"])
    df   = df.dropna(subset=["close"]).sort_values("date").tail(N_ROWS).reset_index(drop=True)
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna().values
    return log_ret.astype(np.float64)


def split(y: np.ndarray):
    n      = len(y)
    t, v   = int(n * 0.70), int(n * 0.85)
    return y[:t], y[t:v], y[v:]


os.makedirs(SAVE_DIR, exist_ok=True)
all_scores: list[pd.DataFrame] = []

for ticker in TICKERS:
    print(f"\n{'='*55}")
    print(f"  {ticker}")
    print(f"{'='*55}")

    y              = load_log_returns(ticker)
    y_train, y_val, y_test = split(y)

    print(f"  samples  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    model = ArimaModel(auto_order=True, p_max=4, q_max=4)
    model.fit(None, y_train, X_val=None, y_val=y_val, verbose=True)

    sc = model.scores(None, y_test, model_name=ticker)
    all_scores.append(sc)
    print(f"\n  Test metrics:")
    print(sc.to_string())

    save_path = os.path.join(SAVE_DIR, f"{ticker}_arima.pkl")
    model.save(save_path)
    print(f"\n  Model saved → {save_path}")

print(f"\n{'='*55}")
print("  SUMMARY")
print(f"{'='*55}")
print(pd.concat(all_scores).to_string())
