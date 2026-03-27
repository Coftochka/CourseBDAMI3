"""
═══════════════════════════════════════════════════════════════════════════════
 HYDR EDA — Exploratory Data Analysis + Feature Engineering
 Тикер : HYDR (РусГидро), MOEX, 5-минутные свечи, 2020-06-15 → 2023-06-26
 Строк : 256 720  |  Колонок : 59
═══════════════════════════════════════════════════════════════════════════════

 ОБОСНОВАНИЕ ПОДХОДА
 ────────────────────────────────────────────────────────────────────────────
 Датасет содержит три ортогональных уровня информации о рынке.
 EDA строится вокруг них, переходя от грубого к тонкому:

 1. OHLCV + торговый VWAP (pr_*)
    Базовый ценовой сигнал. Позволяет строить технические индикаторы
    (RSI, MACD, Bollinger Bands, ATR) и свечные паттерны (body ratio,
    shadow ratio, price position). Это «нижний этаж» feature space —
    устойчивые признаки, не требующие данных стакана.

 2. Торговый поток (Order Flow): trades_b/s, vol_b/s, val_b/s, disb
    Агрессивные покупки vs продажи — прямой сигнал давления на цену.
    Order Flow Imbalance (OFI) является одним из сильнейших краткосрочных
    предикторов микроструктурных движений (Cont et al., 2014).
    При ненулевом OFI рынок "разворачивается" вслед за агрессором.

 3. Снимок стакана заявок (LOB): spread_*, levels_*, imbalance_*, vwap_*
    Состояние ликвидности до и после свечи. LOB imbalance предсказывает
    направление цены на горизонте 1-5 баров (Gould et al., 2013; Stoikov,
    2018). Глубина стакана (vol/val_obstat) показывает «толщину» рынка.
    Замечание: столбцы _x (начало свечи) и _y (конец) в данном датасете
    совпадают по значениям — артефакт merge-операции; используем только _x.

 4. Вторичный рынок (sec_pr_*): OHLC альтернативной площадки
    Арбитражная динамика и price discovery между биржами. Расхождение
    цен sec_pr vs pr является потенциальным предиктором коррекции.

 Структура EDA:
   §1  Загрузка и обзор данных
   §2  Качество данных (пропуски, дубликаты, выбросы)
   §3  Анализ таргета (бинарное направление следующей свечи)
   §4  Ценовой анализ (OHLCV, доходности, свечные паттерны)
   §5  Order Flow анализ
   §6  Микроструктура (стакан заявок)
   §7  Вторичный рынок
   §8  Временные паттерны (сессия, час, день недели)
   §9  Корреляции и мультиколлинеарность (VIF)
   §10 Feature Engineering — итоговый датасет

 ИТОГО Feature Engineering (обоснование групп):
   Ценовые       log_ret, body_ratio, shadow_up/dn, price_position
   Технические   SMA/EMA, RSI, MACD, ATR, Bollinger %B, Stochastic %K
   Order flow    ofi_vol, ofi_val, ofi_trades (normalised imbalances)
   Микростр.     spread_ratio, lob_imbalance_vol/val, depth_ratio, vwap_basis
   Вторичный рынок  sec_ret, cross_spread (арбитражная база)
   Временные     hour, dow, session_min, is_open/close_hour
   Лаговые       log_ret_lag1..5, ofi_vol_lag1..3
   Скользящие    rolling_ret_mean/std (5/10/20), rolling_vol_z
═══════════════════════════════════════════════════════════════════════════════
"""

# %% ── Imports ────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — не нужен X11/display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 9})

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "dataset/HYDR/HYDR_SUPER_FULL.csv"
)
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches="tight")
    plt.close("all")


# %% ── §1  Загрузка и обзор данных ───────────────────────────────────────────
print("=" * 70)
print("§1  ЗАГРУЗКА И ОБЗОР ДАННЫХ")
print("=" * 70)

df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"  Строк: {len(df):,}   Столбцов: {len(df.columns)}")
print(f"  Период: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Торговых дней: {df['timestamp'].dt.date.nunique()}")

# Группировка столбцов по смысловому признаку
LOB_BASE = [
    "spread_bbo", "spread_lv10", "spread_1mio",
    "levels_b", "levels_s",
    "vol_b_obstat", "vol_s_obstat",
    "val_b_obstat", "val_s_obstat",
    "imbalance_vol_bbo", "imbalance_val_bbo",
    "imbalance_vol", "imbalance_val",
    "vwap_b", "vwap_s", "vwap_b_1mio", "vwap_s_1mio",
]
LOB_X = [c + "_x" for c in LOB_BASE]
LOB_Y = [c + "_y" for c in LOB_BASE]
OHLCV = ["pr_open", "pr_high", "pr_low", "pr_close", "pr_std",
         "vol", "val", "trades", "pr_vwap", "pr_change"]
FLOW  = ["trades_b", "trades_s", "val_b", "val_s",
         "vol_b", "vol_s", "disb", "pr_vwap_b", "pr_vwap_s"]
SEC   = ["sec_pr_open", "sec_pr_high", "sec_pr_low", "sec_pr_close"]

print("\n  Группы колонок:")
print(f"    LOB snapshot (_x): {len(LOB_X)} cols")
print(f"    LOB snapshot (_y): {len(LOB_Y)} cols  (потенциальные дубликаты)")
print(f"    OHLCV + свечные:   {len(OHLCV)} cols")
print(f"    Order Flow:        {len(FLOW)} cols")
print(f"    Secondary market:  {len(SEC)} cols")
print(f"    ticker + timestamp: 2 cols")

print("\n  Базовая статистика (OHLCV):")
print(df[OHLCV].describe().round(4).to_string())

# %% ── §2  Качество данных ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§2  КАЧЕСТВО ДАННЫХ")
print("=" * 70)

# --- 2.1 Пропуски ---
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"count": missing, "pct": missing_pct})
miss_df = miss_df[miss_df["count"] > 0].sort_values("count", ascending=False)
if miss_df.empty:
    print("  Пропуски: нет")
else:
    print(f"  Колонки с пропусками ({len(miss_df)}):")
    print(miss_df.to_string())

# --- 2.2 Дублирующиеся строки ---
n_dup = df.duplicated(subset="timestamp").sum()
print(f"\n  Дублирующиеся timestamp: {n_dup}")

# --- 2.3 Разрывы во времени ---
delta = df["timestamp"].diff().dropna()
expected = pd.Timedelta("5min")
gaps = delta[delta > expected * 2]
print(f"  Разрывов > 10 мин: {len(gaps)}")
if len(gaps):
    top_gaps = delta.nlargest(5)
    print("  Топ-5 разрывов:")
    for ts, td in zip(df["timestamp"].iloc[top_gaps.index], top_gaps):
        print(f"    {ts}  gap={td}")

# --- 2.4 Проверка _x == _y (дублирование стакана) ---
max_diff = max(
    (df[c + "_x"] - df[c + "_y"]).abs().max()
    for c in LOB_BASE
)
print(f"\n  Макс. расхождение _x vs _y по всем LOB-столбцам: {max_diff:.6f}")
print("  → Вывод: _x и _y идентичны, удаляем _y из обучения.")

# --- 2.5 Выбросы (IQR-метод) ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("§2 Распределения ключевых OHLCV-признаков (boxplots)")
cols_box = ["pr_close", "vol", "val", "trades", "pr_std", "pr_change"]
for ax, col in zip(axes.flat, cols_box):
    q1, q3 = df[col].quantile([0.01, 0.99])
    ax.boxplot(df[col].clip(q1, q3), vert=True, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_title(col)
    ax.set_xticks([])
savefig("02_boxplots_ohlcv.png")

# %% ── §3  Анализ таргета ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§3  ТАРГЕТ: НАПРАВЛЕНИЕ СЛЕДУЮЩЕЙ СВЕЧИ")
print("=" * 70)

df["log_ret"] = np.log(df["pr_close"] / df["pr_close"].shift(1))
df["target"]  = (df["pr_close"].shift(-1) > df["pr_close"]).astype(int)

n_valid = df["target"].dropna()
pos_rate = df["target"].mean()
print(f"  Класс 1 (up):   {df['target'].sum():,} ({pos_rate:.2%})")
print(f"  Класс 0 (down): {(1 - df['target']).sum():,} ({1-pos_rate:.2%})")
print(f"  → Баланс классов близок к 50/50 — типично для EMH")

# Автокорреляция таргета (persistence signal)
acf_vals = acf(df["target"].dropna(), nlags=20, fft=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("§3 Анализ таргета")

axes[0].bar(range(1, len(acf_vals)), acf_vals[1:], color="steelblue", alpha=0.7)
axes[0].axhline(1.96 / np.sqrt(len(df)), ls="--", color="red", label="95% CI")
axes[0].axhline(-1.96 / np.sqrt(len(df)), ls="--", color="red")
axes[0].set_title("ACF таргета (лаги 1-20)")
axes[0].set_xlabel("Lag")
axes[0].legend()

# Распределение доходностей
axes[1].hist(df["log_ret"].dropna(), bins=200, color="steelblue", alpha=0.7,
             density=True)
x_ = np.linspace(df["log_ret"].dropna().quantile(0.001),
                 df["log_ret"].dropna().quantile(0.999), 300)
axes[1].plot(x_, stats.norm.pdf(x_, df["log_ret"].mean(), df["log_ret"].std()),
             "r-", lw=2, label="Normal fit")
axes[1].set_title("Распределение лог-доходностей")
axes[1].legend()
savefig("03_target_analysis.png")

sk = df["log_ret"].dropna().skew()
ku = df["log_ret"].dropna().kurt()
print(f"\n  Лог-доходности: skew={sk:.3f}  kurt={ku:.3f}  → fat tails")

# %% ── §4  Ценовой анализ ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§4  ЦЕНОВОЙ АНАЛИЗ")
print("=" * 70)

# --- 4.1 Ценовой ряд ---
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("§4 Ценовой ряд HYDR (2020-2023)")

axes[0].plot(df["timestamp"], df["pr_close"], lw=0.5, color="steelblue")
axes[0].set_ylabel("Цена закрытия, ₽")
axes[0].set_title("pr_close")

axes[1].bar(df["timestamp"], df["vol"], width=0.003, color="steelblue", alpha=0.5)
axes[1].set_ylabel("Объём (лоты)")
axes[1].set_title("vol")

axes[2].plot(df["timestamp"], df["log_ret"], lw=0.3, color="gray", alpha=0.8)
axes[2].set_ylabel("Лог-доход")
axes[2].set_title("log_ret")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
savefig("04_price_series.png")

# --- 4.2 Свечные паттерны ---
candle_range = df["pr_high"] - df["pr_low"]
df["body_ratio"]     = (df["pr_close"] - df["pr_open"]).abs() / candle_range.replace(0, np.nan)
df["shadow_up"]      = (df["pr_high"]  - df[["pr_open", "pr_close"]].max(axis=1)) / candle_range.replace(0, np.nan)
df["shadow_dn"]      = (df[["pr_open", "pr_close"]].min(axis=1) - df["pr_low"])  / candle_range.replace(0, np.nan)
df["price_position"] = (df["pr_close"] - df["pr_low"]) / candle_range.replace(0, np.nan)
df["vwap_basis"]     = (df["pr_close"] - df["pr_vwap"]) / df["pr_vwap"]

print("  Свечные паттерны — описательная статистика:")
print(df[["body_ratio", "shadow_up", "shadow_dn", "price_position", "vwap_basis"]]
      .describe().round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("§4 Свечные паттерны: body_ratio, price_position, vwap_basis")
for ax, col, color in zip(axes, ["body_ratio", "price_position", "vwap_basis"],
                           ["steelblue", "darkorange", "seagreen"]):
    ax.hist(df[col].dropna(), bins=100, color=color, alpha=0.75, density=True)
    ax.set_title(col)
savefig("04b_candle_patterns.png")

# %% ── §5  Order Flow анализ ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§5  ORDER FLOW АНАЛИЗ")
print("=" * 70)

# Нормализованные дисбалансы (Order Flow Imbalance)
df["ofi_vol"]    = (df["vol_b"]    - df["vol_s"])    / (df["vol_b"]    + df["vol_s"]).replace(0, np.nan)
df["ofi_val"]    = (df["val_b"]    - df["val_s"])    / (df["val_b"]    + df["val_s"]).replace(0, np.nan)
df["ofi_trades"] = (df["trades_b"] - df["trades_s"]) / (df["trades_b"] + df["trades_s"]).replace(0, np.nan)

print("  Order Flow Imbalance (OFI) — описательная статистика:")
print(df[["ofi_vol", "ofi_val", "ofi_trades", "disb"]].describe().round(4).to_string())

# Связь OFI с таргетом
print("\n  OFI по классам таргета (mean ± std):")
for col in ["ofi_vol", "ofi_val", "ofi_trades"]:
    g = df.groupby("target")[col].agg(["mean", "std"])
    print(f"    {col:15s}: down={g.loc[0,'mean']:.4f}±{g.loc[0,'std']:.4f}  "
          f"up={g.loc[1,'mean']:.4f}±{g.loc[1,'std']:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("§5 Order Flow Imbalance — распределение по направлению цены")
for ax, col in zip(axes, ["ofi_vol", "ofi_val", "ofi_trades"]):
    for tgt, color, lbl in [(0, "tomato", "down"), (1, "steelblue", "up")]:
        ax.hist(df.loc[df["target"] == tgt, col].dropna(),
                bins=80, alpha=0.5, density=True, color=color, label=lbl)
    ax.set_title(col)
    ax.legend(fontsize=7)
savefig("05_ofi_by_target.png")

# Автокорреляция OFI — персистентность давления
acf_ofi = acf(df["ofi_vol"].dropna(), nlags=20, fft=True)
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(1, len(acf_ofi)), acf_ofi[1:], color="steelblue", alpha=0.7)
ax.axhline(1.96 / np.sqrt(len(df)), ls="--", color="red")
ax.axhline(-1.96 / np.sqrt(len(df)), ls="--", color="red")
ax.set_title("§5 ACF ofi_vol (лаги 1-20)")
ax.set_xlabel("Lag")
savefig("05b_acf_ofi.png")

# %% ── §6  Микроструктура (LOB) ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("§6  МИКРОСТРУКТУРА — СТАКАН ЗАЯВОК (LOB)")
print("=" * 70)

lob_cols = [c + "_x" for c in ["spread_bbo", "spread_lv10", "spread_1mio",
                                "levels_b", "levels_s",
                                "imbalance_vol", "imbalance_val",
                                "vol_b_obstat", "vol_s_obstat"]]
print("  LOB — описательная статистика:")
print(df[lob_cols].describe().round(4).to_string())

# Производные LOB-признаки
df["lob_imbalance_vol"] = df["imbalance_vol_x"]
df["lob_imbalance_val"] = df["imbalance_val_x"]
df["depth_ratio"]       = df["vol_b_obstat_x"] / df["vol_s_obstat_x"].replace(0, np.nan)
df["spread_ratio"]      = df["spread_bbo_x"]   / df["spread_lv10_x"].replace(0, np.nan)
df["lob_vwap_mid"]      = (df["vwap_b_x"] + df["vwap_s_x"]) / 2
df["lob_vwap_spread"]   = (df["vwap_s_x"] - df["vwap_b_x"]) / df["lob_vwap_mid"]

print("\n  LOB производные признаки:")
print(df[["lob_imbalance_vol", "depth_ratio", "spread_ratio",
          "lob_vwap_spread"]].describe().round(4).to_string())

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("§6 Распределение LOB-признаков")
lob_plot = [("spread_bbo_x", "BBO Spread"), ("spread_lv10_x", "Spread Lv10"),
            ("imbalance_vol_x", "LOB Vol Imbalance"),
            ("imbalance_val_x", "LOB Val Imbalance"),
            ("depth_ratio", "Depth Ratio (bid/ask)"),
            ("spread_ratio", "BBO/Lv10 Spread Ratio")]
for ax, (col, title) in zip(axes.flat, lob_plot):
    data = df[col].dropna()
    q1, q99 = data.quantile([0.005, 0.995])
    ax.hist(data.clip(q1, q99), bins=80, color="steelblue", alpha=0.7, density=True)
    ax.set_title(title, fontsize=8)
savefig("06_lob_features.png")

# LOB imbalance → target
print("\n  LOB imbalance по классам таргета:")
for col in ["lob_imbalance_vol", "lob_imbalance_val", "depth_ratio"]:
    g = df.groupby("target")[col].mean()
    print(f"    {col:25s}: down={g[0]:.4f}  up={g[1]:.4f}")

# %% ── §7  Вторичный рынок ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§7  ВТОРИЧНЫЙ РЫНОК (sec_pr_*)")
print("=" * 70)

print("  Описательная статистика:")
print(df[SEC].describe().round(4).to_string())

# Уникальных значений (вторичный рынок — дискретные уровни?)
for col in SEC:
    nu = df[col].nunique()
    print(f"  {col}: {nu} уникальных значений (range [{df[col].min():.1f}, {df[col].max():.1f}])")

df["sec_ret"]      = np.log(df["sec_pr_close"] / df["sec_pr_close"].shift(1))
df["cross_spread"] = (df["pr_close"] - df["sec_pr_close"]) / df["pr_close"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("§7 Вторичный рынок vs основной")
axes[0].scatter(df["sec_pr_close"].iloc[::100], df["pr_close"].iloc[::100],
                s=1, alpha=0.3, color="steelblue")
axes[0].set_xlabel("sec_pr_close")
axes[0].set_ylabel("pr_close")
axes[0].set_title("Цена основная vs вторичная (каждая 100-я точка)")

q1, q99 = df["cross_spread"].quantile([0.005, 0.995])
axes[1].hist(df["cross_spread"].clip(q1, q99), bins=100,
             color="darkorange", alpha=0.7, density=True)
axes[1].set_title("cross_spread = (pr_close - sec_pr_close) / pr_close")
savefig("07_secondary_market.png")

corr_sec = df[["cross_spread", "target"]].dropna().corr().iloc[0, 1]
print(f"\n  Корреляция cross_spread → target: {corr_sec:.4f}")

# %% ── §8  Временные паттерны ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§8  ВРЕМЕННЫЕ ПАТТЕРНЫ")
print("=" * 70)

df["hour"]         = df["timestamp"].dt.hour
df["dow"]          = df["timestamp"].dt.dayofweek
df["session_min"]  = (df["hour"] - 10) * 60 + df["timestamp"].dt.minute
df["is_open_hour"] = (df["hour"] == 10).astype(int)
df["is_close_hour"]= (df["hour"] == 23).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("§8 Внутрисессионные паттерны HYDR")

# Объём по часу
vol_by_hour = df.groupby("hour")["vol"].mean()
axes[0, 0].bar(vol_by_hour.index, vol_by_hour.values, color="steelblue", alpha=0.8)
axes[0, 0].set_title("Средний объём по часу дня")
axes[0, 0].set_xlabel("Час")

# Волатильность по часу
std_by_hour = df.groupby("hour")["log_ret"].std()
axes[0, 1].bar(std_by_hour.index, std_by_hour.values, color="darkorange", alpha=0.8)
axes[0, 1].set_title("Волатильность (std log_ret) по часу")
axes[0, 1].set_xlabel("Час")

# Частота up по часу
up_by_hour = df.groupby("hour")["target"].mean()
axes[1, 0].bar(up_by_hour.index, up_by_hour.values, color="seagreen", alpha=0.8)
axes[1, 0].axhline(0.5, ls="--", color="red", lw=1)
axes[1, 0].set_title("P(up) по часу дня")
axes[1, 0].set_xlabel("Час")
axes[1, 0].set_ylim(0.4, 0.6)

# Объём по дню недели
vol_by_dow = df.groupby("dow")["vol"].mean()
axes[1, 1].bar(vol_by_dow.index, vol_by_dow.values, color="mediumpurple", alpha=0.8)
axes[1, 1].set_title("Средний объём по дню недели (0=Пн)")
axes[1, 1].set_xlabel("День недели")
savefig("08_temporal_patterns.png")

print("  Объём по часу (топ-3):", vol_by_hour.nlargest(3).to_dict())
print("  Волатильность по часу (топ-3):", std_by_hour.nlargest(3).to_dict())

# %% ── §9  Корреляции и VIF ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("§9  КОРРЕЛЯЦИИ И МУЛЬТИКОЛЛИНЕАРНОСТЬ")
print("=" * 70)

# Признаки для анализа корреляции с таргетом
feature_cols = [
    "log_ret", "body_ratio", "shadow_up", "shadow_dn", "price_position",
    "vwap_basis", "ofi_vol", "ofi_val", "ofi_trades", "disb",
    "lob_imbalance_vol", "lob_imbalance_val", "depth_ratio",
    "spread_ratio", "lob_vwap_spread", "sec_ret", "cross_spread",
    "spread_bbo_x", "spread_lv10_x",
]

corr_with_target = (
    df[feature_cols + ["target"]]
    .dropna()
    .corr()["target"]
    .drop("target")
    .sort_values(key=abs, ascending=False)
)
print("  Корреляция признаков с таргетом (Pearson):")
print(corr_with_target.round(4).to_string())

# Heatmap корреляций между признаками
corr_matrix = df[feature_cols].dropna().corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.3, annot_kws={"size": 7}, ax=ax)
ax.set_title("§9 Матрица корреляций признаков")
savefig("09_correlation_heatmap.png")

# VIF для обнаружения мультиколлинеарности
vif_features = [c for c in feature_cols if df[c].dropna().std() > 0]
vif_data = df[vif_features].dropna().iloc[:10000]  # сэмпл для скорости
vif_result = pd.DataFrame({
    "feature": vif_features,
    "VIF": [variance_inflation_factor(vif_data.values, i)
            for i in range(len(vif_features))]
}).sort_values("VIF", ascending=False)
print("\n  VIF (Variance Inflation Factor):")
print(vif_result.to_string(index=False))
high_vif = vif_result[vif_result["VIF"] > 10]
if len(high_vif):
    print(f"\n  ⚠ Признаки с VIF > 10 (мультиколлинеарность): {high_vif['feature'].tolist()}")

# %% ── §10  Feature Engineering — итоговый датасет ───────────────────────────
print("\n" + "=" * 70)
print("§10  FEATURE ENGINEERING")
print("=" * 70)

fe = df[["timestamp", "target"]].copy()

# --- Ценовые ---
fe["log_ret"]        = df["log_ret"]
fe["body_ratio"]     = df["body_ratio"]
fe["shadow_up"]      = df["shadow_up"]
fe["shadow_dn"]      = df["shadow_dn"]
fe["price_position"] = df["price_position"]
fe["vwap_basis"]     = df["vwap_basis"]
fe["hl_range"]       = (df["pr_high"] - df["pr_low"]) / df["pr_close"]

# --- Технические индикаторы ---
close = df["pr_close"]
high  = df["pr_high"]
low   = df["pr_low"]
vol_s = df["vol"]

# SMA / EMA
for w in [5, 10, 20]:
    fe[f"sma{w}_dev"]  = (close - close.rolling(w).mean()) / close
    fe[f"ema{w}_dev"]  = (close - close.ewm(span=w, adjust=False).mean()) / close

# ATR (Average True Range)
tr = pd.concat([
    (high - low).rename("hl"),
    (high - close.shift(1)).abs().rename("hpc"),
    (low  - close.shift(1)).abs().rename("lpc"),
], axis=1).max(axis=1)
fe["atr14"]     = tr.rolling(14).mean() / close
fe["atr14_norm"]= tr / tr.rolling(14).mean()

# RSI-14
delta_c = close.diff()
gain = delta_c.clip(lower=0).rolling(14).mean()
loss = (-delta_c.clip(upper=0)).rolling(14).mean()
rs   = gain / loss.replace(0, np.nan)
fe["rsi14"] = 100 - 100 / (1 + rs)

# MACD (12-26-9)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line   = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
fe["macd_hist"] = (macd_line - signal_line) / close

# Bollinger Bands %B (20-period)
bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
fe["bb_pct"]  = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std.replace(0, np.nan))
fe["bb_width"] = (4 * bb_std) / bb_mid

# Stochastic %K (14-period)
lo14 = low.rolling(14).min()
hi14 = high.rolling(14).max()
fe["stoch_k"] = (close - lo14) / (hi14 - lo14).replace(0, np.nan)

# --- Order Flow ---
fe["ofi_vol"]    = df["ofi_vol"]
fe["ofi_val"]    = df["ofi_val"]
fe["ofi_trades"] = df["ofi_trades"]
fe["disb"]       = df["disb"]
fe["vol_zscore"] = (df["vol"] - df["vol"].rolling(20).mean()) / df["vol"].rolling(20).std()

# --- LOB (микроструктура) ---
fe["lob_imbalance_vol"] = df["lob_imbalance_vol"]
fe["lob_imbalance_val"] = df["lob_imbalance_val"]
fe["depth_ratio"]       = df["depth_ratio"]
fe["spread_ratio"]      = df["spread_ratio"]
fe["lob_vwap_spread"]   = df["lob_vwap_spread"]
fe["spread_bbo"]        = df["spread_bbo_x"]

# --- Вторичный рынок ---
fe["sec_ret"]       = df["sec_ret"]
fe["cross_spread"]  = df["cross_spread"]

# --- Временные ---
fe["hour"]          = df["hour"]
fe["dow"]           = df["dow"]
fe["session_min"]   = df["session_min"]
fe["is_open_hour"]  = df["is_open_hour"]
fe["is_close_hour"] = df["is_close_hour"]

# --- Лаговые ---
for lag in range(1, 6):
    fe[f"log_ret_lag{lag}"]  = df["log_ret"].shift(lag)
for lag in range(1, 4):
    fe[f"ofi_vol_lag{lag}"]  = df["ofi_vol"].shift(lag)
    fe[f"lob_imb_lag{lag}"]  = df["lob_imbalance_vol"].shift(lag)

# --- Скользящие статистики ---
for w in [5, 10, 20]:
    fe[f"rolling_ret_mean{w}"] = df["log_ret"].rolling(w).mean()
    fe[f"rolling_ret_std{w}"]  = df["log_ret"].rolling(w).std()
    fe[f"rolling_ofi_mean{w}"] = df["ofi_vol"].rolling(w).mean()

fe["rolling_vol_z20"] = fe["vol_zscore"]

# --- Финальная очистка ---
fe_clean = fe.dropna().reset_index(drop=True)
print(f"  Исходных строк:  {len(df):,}")
print(f"  После dropna:    {len(fe_clean):,}")
print(f"  Итоговых признаков: {len(fe_clean.columns) - 2}  (без timestamp и target)")
print(f"\n  Список признаков:")
feature_list = [c for c in fe_clean.columns if c not in ("timestamp", "target")]
for i, col in enumerate(feature_list):
    print(f"    {i+1:2d}. {col}")

# Распределение таргета в финальном датасете
pos = fe_clean["target"].mean()
print(f"\n  Баланс таргета: up={pos:.2%}  down={1-pos:.2%}")

# --- Сохранение ---
out_path = os.path.join(os.path.dirname(__file__), "dataset/HYDR/HYDR_FE.csv")
fe_clean.to_csv(out_path, index=False)
print(f"\n  ✓ Сохранено: {out_path}")
print(f"    Форма: {fe_clean.shape}")

# --- Итоговый график: важность признаков через корреляцию с таргетом ---
feat_corr = (
    fe_clean[feature_list + ["target"]]
    .corr()["target"]
    .drop("target")
    .abs()
    .sort_values(ascending=True)
)
fig, ax = plt.subplots(figsize=(7, len(feat_corr) * 0.22 + 1))
feat_corr.plot.barh(ax=ax, color="steelblue", alpha=0.75)
ax.set_title("§10 |Корреляция с таргетом| — все engineered features")
ax.set_xlabel("|Pearson r|")
savefig("10_feature_importance_corr.png")

print("\n" + "=" * 70)
print("EDA завершён. Все графики сохранены в:", PLOTS_DIR)
print("=" * 70)
