"""
Sliding-window inference of Kronos-base over a full MOEX dataset.

For each non-overlapping window the model predicts the next PRED_LEN candles.
Results are saved to a CSV with ground-truth and predicted OHLCV side-by-side.

Output columns:
    timestamp          – start of the predicted candle
    context_end        – last timestamp of the context window
    horizon            – step index within the forecast (1 … PRED_LEN)
    gt_{open,high,low,close,volume,amount}
    pred_{open,high,low,close,volume,amount}
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))
from model import Kronos, KronosTokenizer, KronosPredictor


# ─── Config ───────────────────────────────────────────────────────────────────

TICKER       = "SBER"
DATASET_PATH = os.path.join(
    os.path.dirname(__file__), f"../data/dataset/{TICKER}/{TICKER}_SUPER_FULL.csv"
)

LOOKBACK     = 400   # контекст (≤ 512)
PRED_LEN     = 60    # горизонт прогноза в свечах (60 × 5 мин = 5 часов)
STRIDE       = PRED_LEN  # шаг окна: PRED_LEN → неперекрывающиеся окна

# Батч-инференс: сколько окон обрабатывать за один вызов predict_batch
# Увеличить при большом VRAM; уменьшить если OOM
BATCH_SIZE   = 16

SAMPLE_COUNT = 1     # 1 – быстро; >1 – среднее по sample_count путям (лучше, но дольше)
T            = 0.8
TOP_P        = 0.9

# None – обработать весь датасет; число – только последние N_WINDOWS окон
N_WINDOWS    = None

COL_MAP = {
    "pr_open":  "open",
    "pr_high":  "high",
    "pr_low":   "low",
    "pr_close": "close",
    "vol":      "volume",
    "val":      "amount",
}
OHLCVA = ["open", "high", "low", "close", "volume", "amount"]

OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    f"../data/dataset/{TICKER}/{TICKER}_kronos_predictions.csv"
)

# ─── Load data ────────────────────────────────────────────────────────────────

print(f"Loading {TICKER}...")
df_raw = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
df_raw = df_raw.dropna(subset=list(COL_MAP.keys())).reset_index(drop=True)
df_raw = df_raw.rename(columns=COL_MAP)
df_raw = df_raw[["timestamp"] + OHLCVA]

print(f"  Candles: {len(df_raw)}  |  {df_raw['timestamp'].iloc[0]} → {df_raw['timestamp'].iloc[-1]}")

# Build list of (start_idx, end_idx) for each window
#   context: [start_idx, start_idx + LOOKBACK)
#   target:  [start_idx + LOOKBACK, start_idx + LOOKBACK + PRED_LEN)
window_starts = list(range(0, len(df_raw) - LOOKBACK - PRED_LEN + 1, STRIDE))

if N_WINDOWS is not None:
    window_starts = window_starts[-N_WINDOWS:]

total_windows = len(window_starts)
print(f"  Windows: {total_windows}  (lookback={LOOKBACK}, pred_len={PRED_LEN}, stride={STRIDE})")

# ─── Load model ───────────────────────────────────────────────────────────────

print("\nLoading Kronos-base...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print(f"  Device: {predictor.device}")

# ─── Sliding-window inference ─────────────────────────────────────────────────

records = []
t0 = time.time()

batches = [
    window_starts[i : i + BATCH_SIZE]
    for i in range(0, total_windows, BATCH_SIZE)
]

for batch_starts in tqdm(batches, desc="Batches", unit="batch"):
    df_list         = []
    x_timestamp_list = []
    y_timestamp_list = []
    gt_list          = []
    context_end_list = []

    for s in batch_starts:
        ctx = df_raw.iloc[s : s + LOOKBACK]
        tgt = df_raw.iloc[s + LOOKBACK : s + LOOKBACK + PRED_LEN]

        df_list.append(ctx[OHLCVA].reset_index(drop=True))
        x_timestamp_list.append(ctx["timestamp"].reset_index(drop=True))
        y_timestamp_list.append(tgt["timestamp"].reset_index(drop=True))
        gt_list.append(tgt[OHLCVA].reset_index(drop=True))
        context_end_list.append(ctx["timestamp"].iloc[-1])

    pred_list = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_timestamp_list,
        y_timestamp_list=y_timestamp_list,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
    )

    for i, (pred_df, gt_df, y_ts, ctx_end) in enumerate(
        zip(pred_list, gt_list, y_timestamp_list, context_end_list)
    ):
        for h in range(PRED_LEN):
            row = {
                "timestamp":   y_ts.iloc[h],
                "context_end": ctx_end,
                "horizon":     h + 1,
            }
            for col in OHLCVA:
                row[f"gt_{col}"]   = gt_df[col].iloc[h]
                row[f"pred_{col}"] = pred_df[col].iloc[h]
            records.append(row)

elapsed = time.time() - t0
print(f"\nInference done in {elapsed:.1f}s  ({elapsed/total_windows:.2f}s per window)")

# ─── Save ─────────────────────────────────────────────────────────────────────

result_df = pd.DataFrame(records)
result_df = result_df.sort_values("timestamp").reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
result_df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(result_df)} rows → {OUT_PATH}")

# ─── Aggregate metrics ────────────────────────────────────────────────────────

gt_c   = result_df["gt_close"].values
pred_c = result_df["pred_close"].values

mae  = np.mean(np.abs(pred_c - gt_c))
rmse = np.sqrt(np.mean((pred_c - gt_c) ** 2))
mape = np.mean(np.abs((pred_c - gt_c) / gt_c)) * 100

# Per-horizon metrics
print("\n── Per-horizon close MAE ──")
for h in range(1, PRED_LEN + 1, max(1, PRED_LEN // 10)):
    mask = result_df["horizon"] == h
    h_mae = np.mean(np.abs(
        result_df.loc[mask, "pred_close"].values -
        result_df.loc[mask, "gt_close"].values
    ))
    bar = "█" * int(h_mae / mae * 10)
    print(f"  h={h:3d}  MAE={h_mae:.4f}  {bar}")

print(f"\n── Overall (close price) ──")
print(f"  Windows : {total_windows}")
print(f"  Rows    : {len(result_df)}")
print(f"  MAE     : {mae:.4f}")
print(f"  RMSE    : {rmse:.4f}")
print(f"  MAPE    : {mape:.4f}%")
print(f"  Output  : {OUT_PATH}")
