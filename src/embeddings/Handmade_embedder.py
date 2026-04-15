import numpy as np
from scipy import stats

OPEN = 0; HIGH = 1; LOW = 2; CLOSE = 3; VOLUME = 4
SMA5 = 5; SMA20 = 6; EMA12 = 7; EMA26 = 8; CLOSE_SMA20 = 9
MACD = 10; MACD_SIG = 11; MACD_HIST = 12; RSI14 = 13
BB_PCT = 14; BB_BW = 15; ATR14 = 16; OBV = 17

FEATURE_NAMES = [
    "slope",            
    "net_move",         
    "frac_above_mean",  
    "price_range",      
    "daily_vol",        
    "skewness",         
    "kurtosis",         
    "price_vol_corr",   
    "vol_activity",     
    "last_rsi",         
    "last_macd_hist",   
    "last_bb_pct",      
    "last_bb_bw",       
    "rsi_slope",        
    "macd_slope",       
]


class HandmadeEmbedder:
    def transform(self, windows: np.ndarray) -> np.ndarray:
        close  = windows[:, :, CLOSE].astype(np.float64)
        volume = windows[:, :, VOLUME].astype(np.float64)
        T = close.shape[1]

        x   = np.arange(T, dtype=np.float64)
        x_c = x - x.mean()
        slope = (x_c * (close - close.mean(axis=1, keepdims=True))).sum(axis=1) / (x_c ** 2).sum()

        net_move = close[:, -1] - close[:, 0]

        frac_above_mean = (close > close.mean(axis=1, keepdims=True)).mean(axis=1)
        price_range = close.max(axis=1) - close.min(axis=1)

        diff_close = np.diff(close, axis=1)
        daily_vol = diff_close.std(axis=1)

        skewness = stats.skew(close, axis=1)
        kurtosis = stats.kurtosis(close, axis=1)

        close_c = close  - close.mean(axis=1, keepdims=True)
        vol_c   = volume - volume.mean(axis=1, keepdims=True)
        denom   = np.sqrt((close_c**2).sum(axis=1) * (vol_c**2).sum(axis=1)) + 1e-8
        price_vol_corr = (close_c * vol_c).sum(axis=1) / denom

        vol_activity = np.diff(volume, axis=1).std(axis=1)

        last_rsi       = windows[:, -1, RSI14  ].astype(np.float64)
        last_macd_hist = windows[:, -1, MACD_HIST].astype(np.float64)
        last_bb_pct    = windows[:, -1, BB_PCT  ].astype(np.float64)
        last_bb_bw     = windows[:, -1, BB_BW   ].astype(np.float64)

        first_rsi       = windows[:, 0, RSI14    ].astype(np.float64)
        first_macd_hist = windows[:, 0, MACD_HIST].astype(np.float64)
        rsi_slope  = last_rsi       - first_rsi
        macd_slope = last_macd_hist - first_macd_hist

        emb = np.stack([
            slope,
            net_move,
            frac_above_mean,
            price_range,
            daily_vol,
            skewness,
            kurtosis,
            price_vol_corr,
            vol_activity,
            last_rsi,
            last_macd_hist,
            last_bb_pct,
            last_bb_bw,
            rsi_slope,
            macd_slope,
        ], axis=1)

        return np.where(np.isfinite(emb), emb, 0.0).astype(np.float32)
