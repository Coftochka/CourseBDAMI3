"""
HandmadeEmbedder — 15 численно стабильных признаков из z-нормированных окон.

Входные окна (N, T, F) приходят после per-window z-нормировки Dataloader.make_windows:
    window = (window - mean_t) / (std_t + eps)
Поэтому close[i, t] ≈ N(0, 1) — и НЕЛЬЗЯ делить на close (≈ 0).
Все признаки ниже вычислены через разности / корреляции / percentile-статистики,
а не через процентные доходности.
"""
import numpy as np
from scipy import stats

OPEN = 0; HIGH = 1; LOW = 2; CLOSE = 3; VOLUME = 4
SMA5 = 5; SMA20 = 6; EMA12 = 7; EMA26 = 8; CLOSE_SMA20 = 9
MACD = 10; MACD_SIG = 11; MACD_HIST = 12; RSI14 = 13
BB_PCT = 14; BB_BW = 15; ATR14 = 16; OBV = 17

FEATURE_NAMES = [
    # --- Trend / Direction ---
    "slope",            # линейный наклон z-norm close (>0 = рост в окне)
    "net_move",         # close[-1] - close[0] в единицах std
    "frac_above_mean",  # доля баров выше среднего (0.5=боковик, 0/1=тренд)
    "price_range",      # max - min z-norm close (амплитуда движения)

    # --- Volatility ---
    "daily_vol",        # std первых разностей close (внутрисвечная нестабильность)
    "skewness",         # асимметрия z-norm close (надёжна при T≥60)
    "kurtosis",         # тяжёлые хвосты

    # --- Volume ---
    "price_vol_corr",   # корреляция цены и объёма (подтверждение тренда объёмом)
    "vol_activity",     # std первых разностей volume (насколько объём прыгает)

    # --- Снимок индикаторов в КОНЦЕ окна (позиция относительно window-mean) ---
    "last_rsi",         # RSI в конце окна (z-norm)
    "last_macd_hist",   # MACD гистограмма в конце
    "last_bb_pct",      # где цена в полосе Боллинджера в конце
    "last_bb_bw",       # ширина полос (= волатильный режим) в конце

    # --- НАПРАВЛЕНИЕ индикаторов внутри окна ---
    "rsi_slope",        # last_rsi - first_rsi: импульс нарастает или затухает?
    "macd_slope",       # last_macd_hist - first_macd_hist
]


class HandmadeEmbedder:
    def transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        windows : (N, T, F)  — z-нормированные окна из Dataloader
        Returns
        -------
        emb : (N, 15)
        """
        close  = windows[:, :, CLOSE].astype(np.float64)
        volume = windows[:, :, VOLUME].astype(np.float64)
        T = close.shape[1]

        # ── Trend ────────────────────────────────────────────────────────────
        x   = np.arange(T, dtype=np.float64)
        x_c = x - x.mean()
        # slope = cov(x, close) / var(x) — без деления на close
        slope = (x_c * (close - close.mean(axis=1, keepdims=True))).sum(axis=1) / (x_c ** 2).sum()

        # разность (не отношение!) конца и начала
        net_move = close[:, -1] - close[:, 0]

        frac_above_mean = (close > close.mean(axis=1, keepdims=True)).mean(axis=1)
        price_range = close.max(axis=1) - close.min(axis=1)

        # ── Volatility ───────────────────────────────────────────────────────
        # std первых разностей — стабильно при close ≈ 0
        diff_close = np.diff(close, axis=1)
        daily_vol = diff_close.std(axis=1)

        skewness = stats.skew(close, axis=1)
        kurtosis = stats.kurtosis(close, axis=1)

        # ── Volume ───────────────────────────────────────────────────────────
        close_c = close  - close.mean(axis=1, keepdims=True)
        vol_c   = volume - volume.mean(axis=1, keepdims=True)
        denom   = np.sqrt((close_c**2).sum(axis=1) * (vol_c**2).sum(axis=1)) + 1e-8
        price_vol_corr = (close_c * vol_c).sum(axis=1) / denom

        # std первых разностей объёма — стабильно при volume ≈ 0
        vol_activity = np.diff(volume, axis=1).std(axis=1)

        # ── Indicator snapshots at END ────────────────────────────────────────
        last_rsi       = windows[:, -1, RSI14  ].astype(np.float64)
        last_macd_hist = windows[:, -1, MACD_HIST].astype(np.float64)
        last_bb_pct    = windows[:, -1, BB_PCT  ].astype(np.float64)
        last_bb_bw     = windows[:, -1, BB_BW   ].astype(np.float64)

        # ── Indicator slopes (direction within window) ────────────────────────
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
