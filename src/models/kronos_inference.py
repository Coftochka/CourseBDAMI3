"""
Zero-shot inference of Kronos-base on MOEX SBER 5-min candles.
Model: NeoQuasar/Kronos-base (102.3M params) + NeoQuasar/Kronos-Tokenizer-base
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))
from model import Kronos, KronosTokenizer, KronosPredictor


# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "../data/dataset/SBER/SBER_SUPER_FULL.csv"
)
TICKER = "SBER"

LOOKBACK = 512       # контекстное окно (макс 512 для base/small)
PRED_LEN = 12        # 60 свечей × 5 мин = 5 торговых часов вперёд
SAMPLE_COUNT = 5     # кол-во параллельных выборок (усредняются)
T = 0.8              # температура
TOP_P = 0.9          # nucleus sampling

COL_MAP = {
    "pr_open":  "open",
    "pr_high":  "high",
    "pr_low":   "low",
    "pr_close": "close",
    "vol":      "volume",
    "val":      "amount",
}

# ─── Load & prepare data ──────────────────────────────────────────────────────

print(f"Loading {TICKER}...")
df_raw = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

# Drop rows with NaN in OHLCV
ohlcv_cols = list(COL_MAP.keys())
df_raw = df_raw.dropna(subset=ohlcv_cols).reset_index(drop=True)

print(f"  Total candles: {len(df_raw)}")
print(f"  Range: {df_raw['timestamp'].iloc[0]} → {df_raw['timestamp'].iloc[-1]}")

# Take a slice: lookback history + pred_len ground truth
# Use the last portion of the dataset for evaluation
start_idx = len(df_raw) - LOOKBACK - PRED_LEN - 1
if start_idx < 0:
    raise ValueError(f"Not enough data: need {LOOKBACK + PRED_LEN}, have {len(df_raw)}")

slice_df = df_raw.iloc[start_idx : start_idx + LOOKBACK + PRED_LEN].copy()

# Rename columns to Kronos format
slice_df = slice_df.rename(columns=COL_MAP)
slice_df = slice_df[["timestamp", "open", "high", "low", "close", "volume", "amount"]]

x_df        = slice_df.iloc[:LOOKBACK][["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = slice_df.iloc[:LOOKBACK]["timestamp"].reset_index(drop=True)
y_timestamp = slice_df.iloc[LOOKBACK:LOOKBACK + PRED_LEN]["timestamp"].reset_index(drop=True)
gt_df       = slice_df.iloc[LOOKBACK:LOOKBACK + PRED_LEN][["open", "high", "low", "close", "volume", "amount"]]

print(f"\n  Context window: {x_timestamp.iloc[0]} → {x_timestamp.iloc[-1]}")
print(f"  Prediction window: {y_timestamp.iloc[0]} → {y_timestamp.iloc[-1]}")

# ─── Load model ───────────────────────────────────────────────────────────────

print("\nLoading Kronos-base + Kronos-Tokenizer-base from HuggingFace...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

predictor = KronosPredictor(model, tokenizer, max_context=512)
print(f"  Device: {predictor.device}")

# ─── Inference ────────────────────────────────────────────────────────────────

print(f"\nRunning inference (lookback={LOOKBACK}, pred_len={PRED_LEN}, sample_count={SAMPLE_COUNT})...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=T,
    top_p=TOP_P,
    sample_count=SAMPLE_COUNT,
    verbose=True,
)

# ─── Metrics ──────────────────────────────────────────────────────────────────

gt_close   = gt_df["close"].values
pred_close = pred_df["close"].values

mae  = np.mean(np.abs(pred_close - gt_close))
rmse = np.sqrt(np.mean((pred_close - gt_close) ** 2))
mape = np.mean(np.abs((pred_close - gt_close) / gt_close)) * 100

# Directional accuracy: did we predict the correct direction of price change?
gt_direction   = np.sign(gt_close[1:] - gt_close[:-1])
pred_direction = np.sign(pred_close[1:] - pred_close[:-1])
dir_acc = np.mean(gt_direction == pred_direction) * 100

# Directional accuracy vs last known close
last_close = x_df["close"].iloc[-1]
gt_dir_vs_last   = np.sign(gt_close - last_close)
pred_dir_vs_last = np.sign(pred_close - last_close)
dir_acc_vs_last  = np.mean(gt_dir_vs_last == pred_dir_vs_last) * 100

print("\n" + "=" * 50)
print(f"  Ticker:     {TICKER}")
print(f"  Pred len:   {PRED_LEN} candles (5-min)")
print(f"  MAE:        {mae:.4f}")
print(f"  RMSE:       {rmse:.4f}")
print(f"  MAPE:       {mape:.4f}%")
print(f"  Dir Acc (step-to-step):  {dir_acc:.1f}%")
print(f"  Dir Acc (vs last close): {dir_acc_vs_last:.1f}%")
print("=" * 50)

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.suptitle(f"Kronos-base zero-shot forecast — {TICKER} (5-min)", fontsize=14, fontweight="bold")

# --- Close price ---
ax1 = axes[0]

# Tail of context window for visual continuity
context_tail = slice_df.iloc[LOOKBACK - 60:LOOKBACK]
ax1.plot(context_tail["timestamp"], context_tail["close"],
         color="steelblue", linewidth=1.5, label="History (last 60 bars)")

ax1.plot(gt_df.index.map(lambda i: y_timestamp.iloc[i - gt_df.index[0]]),
         gt_close, color="green", linewidth=1.8, label="Ground Truth")

ax1.plot(gt_df.index.map(lambda i: y_timestamp.iloc[i - gt_df.index[0]]),
         pred_close, color="red", linewidth=1.8, linestyle="--", label="Kronos-base Forecast")

ax1.axvline(x=y_timestamp.iloc[0], color="gray", linestyle=":", linewidth=1)
ax1.set_ylabel("Close Price, ₽", fontsize=12)
ax1.set_title(
    f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  "
    f"DirAcc(step)={dir_acc:.1f}%  DirAcc(vs_last)={dir_acc_vs_last:.1f}%",
    fontsize=10,
)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)

# --- Volume ---
ax2 = axes[1]
gt_vol   = gt_df["volume"].values
pred_vol = pred_df["volume"].values
ts_arr   = [y_timestamp.iloc[i] for i in range(PRED_LEN)]

ax2.bar(ts_arr, gt_vol,   color="green", alpha=0.5, label="GT Volume",   width=0.003)
ax2.bar(ts_arr, pred_vol, color="red",   alpha=0.5, label="Pred Volume", width=0.003)
ax2.set_ylabel("Volume", fontsize=12)
ax2.set_xlabel("Time", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), f"kronos_forecast_{TICKER}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
plt.show()
