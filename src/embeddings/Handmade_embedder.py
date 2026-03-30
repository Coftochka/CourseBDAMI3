import numpy as np
from scipy import stats

OPEN = 0
HIGH = 1
LOW = 2
CLOSE = 3
VOLUME = 4
SMA5 = 5
SMA20 = 6
EMA12 = 7
EMA26 = 8
CLOSE_SMA20 = 9
MACD = 10
MACD_SIG = 11
MACD_HIST = 12
RSI14 = 13
BB_PCT = 14
BB_BW = 15
ATR14 = 16
OBV = 17


class HandmadeEmbedder:
    def transform(self, windows: np.ndarray) -> np.ndarray:
        close  = windows[:, :, CLOSE]
        volume = windows[:, :, VOLUME]

        x = np.arange(windows.shape[1], dtype=np.float32)
        x_c = x - x.mean()
        slope = (x_c * (close - close.mean(axis=1, keepdims=True))).sum(axis=1) / (x_c ** 2).sum()
        slope_norm = slope / (close.mean(axis=1) + 1e-8)

        window_return = close[:, -1] / (close[:, 0] + 1e-8) - 1
        frac_above_mean = (close > close.mean(axis=1, keepdims=True)).mean(axis=1)

        returns = np.diff(close, axis=1) / (close[:, :-1] + 1e-8)
        realized_vol = returns.std(axis=1)
        mean_bb_bw = windows[:, :, BB_BW].mean(axis=1)
        mean_atr_norm = (windows[:, :, ATR14] / (close + 1e-8)).mean(axis=1)

        mean_rsi = windows[:, :, RSI14].mean(axis=1)
        last_rsi = windows[:, -1, RSI14]
        mean_macd_hist = windows[:, :, MACD_HIST].mean(axis=1)
        last_macd_hist = windows[:, -1, MACD_HIST]

        mean_bb_pct = windows[:, :, BB_PCT].mean(axis=1)
        last_bb_pct = windows[:, -1, BB_PCT]
        mean_close_sma20 = windows[:, :, CLOSE_SMA20].mean(axis=1)

        skewness = stats.skew(close, axis=1)
        kurtosis = stats.kurtosis(close, axis=1)
        price_range = close.max(axis=1) - close.min(axis=1)

        vol_std_norm = volume.std(axis=1) / (volume.mean(axis=1) + 1e-8)
        close_c = close - close.mean(axis=1, keepdims=True)
        vol_c   = volume - volume.mean(axis=1, keepdims=True)
        price_vol_corr = (close_c * vol_c).sum(axis=1) / (
            np.sqrt((close_c ** 2).sum(axis=1) * (vol_c ** 2).sum(axis=1)) + 1e-8
        )

        return np.stack([
            slope_norm,
            window_return,
            frac_above_mean,
            realized_vol,
            mean_bb_bw,
            mean_atr_norm,
            mean_rsi,
            last_rsi,
            mean_macd_hist,
            last_macd_hist,
            mean_bb_pct,
            last_bb_pct,
            mean_close_sma20,
            skewness,
            kurtosis,
            price_range,
            vol_std_norm,
            price_vol_corr,
        ], axis=1).astype(np.float32)