"""
ComparisonTable — collects per-cluster, per-model metrics and renders
a pivot summary across the whole experiment.

Usage
-----
    from src.evaluation.compare import ComparisonTable
    from src.evaluation.metrics import regression_metrics

    table = ComparisonTable()

    for cluster_id, model_name, y_true, y_pred in results:
        table.add(
            regression_metrics(y_true, y_pred, model_name, cluster=cluster_id)
        )

    table.summary()          # → wide DataFrame: rows=models, cols=metrics×cluster
    table.save("results/exp_001/metrics.csv")
    table.plot_metric("mae") # bar chart comparing models across clusters
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ComparisonTable:

    def __init__(self):
        self._rows: list[pd.DataFrame] = []

    # ── add results ───────────────────────────────────────────────────────────

    def add(self, metrics_df: pd.DataFrame):
        """
        Append a metrics row produced by src.evaluation.metrics.regression_metrics().
        metrics_df must have index = model_name and optional column "cluster".
        """
        self._rows.append(metrics_df)

    def raw(self) -> pd.DataFrame:
        """Full long-form table: one row per (model, cluster) pair."""
        if not self._rows:
            return pd.DataFrame()
        return pd.concat(self._rows)

    # ── aggregation ───────────────────────────────────────────────────────────

    def summary(
        self,
        metric: str = "mae",
        aggfunc: str = "mean",
    ) -> pd.DataFrame:
        """
        Pivot: rows = models, columns = clusters.
        Useful for a quick overview of one metric across all clusters.
        """
        df = self.raw()
        if "cluster" not in df.columns:
            return df[[metric]]
        return df.pivot_table(values=metric, index=df.index, columns="cluster", aggfunc=aggfunc)

    def rank(self, metric: str = "mae", ascending: bool = True) -> pd.DataFrame:
        """
        Average metric across clusters, sorted best→worst.
        ascending=True  → lower is better (MAE, RMSE, MAPE)
        ascending=False → higher is better (R², IC, dir_accuracy)
        """
        df = self.raw()
        avg = df.groupby(df.index)[metric].mean().sort_values(ascending=ascending)
        return avg.rename("avg_" + metric).to_frame()

    # ── I/O ──────────────────────────────────────────────────────────────────

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

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot_metric(
        self,
        metric: str = "mae",
        figsize: tuple = (10, 4),
        title: str | None = None,
    ):
        """
        Bar chart: metric per cluster, grouped by model.
        Works only when "cluster" column is present.
        """
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
