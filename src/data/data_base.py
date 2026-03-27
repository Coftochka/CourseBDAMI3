from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# корень папки с данными: moex_candles лежит рядом с src/
DATA_ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parents[2] / "src" / "data" / "moex_candles"))

from enum import Enum
import pandas as pd


class Interval(Enum):
    Daily  = "daily"    # дневные свечи
    Hourly = "hourly"   # часовые свечи

    def folder(self) -> Path:
        return DATA_ROOT / self.value

    def load(self, ticker: str) -> pd.DataFrame:
        path = self.folder() / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


class Dataset:
    """
    Загружает свечи для одного тикера и делит на train/val/test
    по единым временным порогам (не по индексу).

    Пример:
        ds = Dataset("SBER", Interval.Daily, val_ratio=0.15, test_ratio=0.15)
        train_df = ds.train
        val_df   = ds.val
        test_df  = ds.test
    """

    def __init__(
        self,
        ticker: str,
        interval: Interval = Interval.Daily,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        if val_ratio + test_ratio >= 1.0:
            raise ValueError(f"val_ratio + test_ratio должны быть < 1, получено {val_ratio + test_ratio}")

        self.ticker   = ticker
        self.interval = interval

        df = interval.load(ticker)
        self._split(df, val_ratio, test_ratio)

    def _split(self, df: pd.DataFrame, val_ratio: float, test_ratio: float):
        """Делит по временным порогам, а не по индексу."""
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        total = (t_max - t_min).total_seconds()

        t_val  = t_min + pd.Timedelta(seconds=total * (1 - val_ratio - test_ratio))
        t_test = t_min + pd.Timedelta(seconds=total * (1 - test_ratio))

        self.train = df[df["timestamp"] <  t_val ].reset_index(drop=True)
        self.val   = df[(df["timestamp"] >= t_val) & (df["timestamp"] < t_test)].reset_index(drop=True)
        self.test  = df[df["timestamp"] >= t_test].reset_index(drop=True)

        self.t_val  = t_val
        self.t_test = t_test

    def apply_transform(self, func):
        """Применяет функцию ко всем трём сплитам."""
        self.train = func(self.train)
        self.val   = func(self.val)
        self.test  = func(self.test)
        return self

    def info(self):
        print(f"Ticker:   {self.ticker}  |  interval: {self.interval.value}")
        print(f"Train:    {self.train['timestamp'].min().date()}  →  {self.train['timestamp'].max().date()}  ({len(self.train)} строк)")
        print(f"Val:      {self.val['timestamp'].min().date()}  →  {self.val['timestamp'].max().date()}  ({len(self.val)} строк)")
        print(f"Test:     {self.test['timestamp'].min().date()}  →  {self.test['timestamp'].max().date()}  ({len(self.test)} строк)")


class MultiDataset:
    """
    Загружает несколько тикеров с едиными временными границами split'а.
    Гарантирует что train/val/test у всех тикеров синхронизированы по времени.

    Пример:
        mds = MultiDataset(["SBER", "GAZP", "LKOH"], Interval.Hourly)
        mds.info()

        ticker_to_id = mds.ticker_to_id
        train_dict   = mds.train   # {ticker: df}
    """

    def __init__(
        self,
        tickers: list[str],
        interval: Interval = Interval.Daily,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.tickers      = tickers
        self.interval     = interval
        self.ticker_to_id = {t: i for i, t in enumerate(tickers)}

        # загружаем все df
        raw: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                raw[t] = interval.load(t)
            except FileNotFoundError as e:
                print(f"  ⚠ {e}")

        # единые границы: пересечение временных диапазонов
        t_min = max(df["timestamp"].min() for df in raw.values())
        t_max = min(df["timestamp"].max() for df in raw.values())
        total = (t_max - t_min).total_seconds()

        self.t_val  = t_min + pd.Timedelta(seconds=total * (1 - val_ratio - test_ratio))
        self.t_test = t_min + pd.Timedelta(seconds=total * (1 - test_ratio))
        self.t_min  = t_min
        self.t_max  = t_max

        # режем каждый df по единым порогам
        self.train: dict[str, pd.DataFrame] = {}
        self.val:   dict[str, pd.DataFrame] = {}
        self.test:  dict[str, pd.DataFrame] = {}

        for t, df in raw.items():
            df = df[(df["timestamp"] >= t_min) & (df["timestamp"] <= t_max)]
            self.train[t] = df[df["timestamp"] <  self.t_val ].reset_index(drop=True)
            self.val[t]   = df[(df["timestamp"] >= self.t_val) & (df["timestamp"] < self.t_test)].reset_index(drop=True)
            self.test[t]  = df[df["timestamp"] >= self.t_test].reset_index(drop=True)

    def info(self):
        print(f"Interval: {self.interval.value}  |  tickers: {len(self.tickers)}")
        print(f"Train:  {self.t_min.date()}  →  {self.t_val.date()}")
        print(f"Val:    {self.t_val.date()}  →  {self.t_test.date()}")
        print(f"Test:   {self.t_test.date()}  →  {self.t_max.date()}")
        print()
        for t in self.tickers:
            if t in self.train:
                print(f"  {t:10s}  train={len(self.train[t]):>6}  val={len(self.val[t]):>5}  test={len(self.test[t]):>5}")

    def available_tickers(self) -> list[str]:
        """Список тикеров для которых нашёлся файл."""
        return list(self.train.keys())
