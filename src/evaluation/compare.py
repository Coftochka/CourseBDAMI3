from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ComparisonTable:

    def __init__(self):
        self._rows: list[pd.DataFrame] = []


    def add(self, metrics_df: pd.DataFrame):
        self._rows.append(metrics_df)

    def raw(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.concat(self._rows)


    def summary(
        self,
        metric: str = "mae",
        aggfunc: str = "mean",
    ) -> pd.DataFrame:
        df = self.raw()
        if "cluster" not in df.columns:
            return df[[metric]]
        return df.pivot_table(values=metric, index=df.index, columns="cluster", aggfunc=aggfunc)

    def rank(self, metric: str = "mae", ascending: bool = True) -> pd.DataFrame:
        df = self.raw()
        avg = df.groupby(df.index)[metric].mean().sort_values(ascending=ascending)
        return avg.rename("avg_" + metric).to_frame()


    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.raw().to_csv(path)
        print(f"Metrics saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ComparisonTable":
        table = cls()
        df = pd.read_csv(path, index_col=0)
        for idx in df.index.unique():
            table.add(df.loc[[idx]])
        return table


    def plot_metric(
        self,
        metric: str = "mae",
        figsize: tuple = (10, 4),
        title: str | None = None,
    ):
        df = self.raw()

        if "cluster" not in df.columns:
            df[[metric]].plot.bar(figsize=figsize, legend=False)
            plt.title(title or metric.upper())
            plt.tight_layout()
            plt.show()
            return

        pivot = df.pivot_table(values=metric, index=df.index, columns="cluster")
        pivot.plot.bar(figsize=figsize)
        plt.title(title or f"{metric.upper()} by cluster")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
