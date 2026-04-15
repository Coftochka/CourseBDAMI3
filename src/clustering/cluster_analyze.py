
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

_BG = "#ffffff"
_PANEL = "#f6f8fa"
_GRID = "#d0d7de"
_TEXT = "#1f2328"
_ALPHA_FILL = 0.20
_ALPHA_POINT = 0.65
_NOISE_COLOR = "#8c959f"
_PALETTE = "tab10"

_FONT_TITLE = 14
_FONT_LABEL = 11
_FONT_TICK = 9


def _style(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(_BG)


def _panel(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_TEXT, labelsize=_FONT_TICK)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(color=_GRID, linewidth=0.5, alpha=0.8)


def _cluster_palette(n: int) -> list:
    cmap = plt.cm.get_cmap(_PALETTE, max(n, 1))
    return [cmap(i) for i in range(n)]



class ClusterAnalyzer:
    def __init__(
        self,
        labels: np.ndarray,
        *,
        windows: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        timestamps=None,
        tickers=None,
        close_idx: int = 3,
        feature_names: Optional[list[str]] = None,
        cluster_names: Optional[dict[int, str]] = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=int)
        self.windows = windows
        self.embeddings = embeddings
        self.timestamps = pd.to_datetime(timestamps) if timestamps is not None else None
        self.tickers = np.asarray(tickers, dtype=str) if tickers is not None else None
        self.close_idx = close_idx
        self.feature_names = feature_names
        self._custom_names: dict[int, str] = cluster_names or {}

        self._valid = self.labels != -1
        self._clusters: list[int] = sorted(
            np.unique(self.labels[self._valid]).tolist()
        )
        n_c = len(self._clusters)
        palette = _cluster_palette(n_c)
        self._colors: dict[int, tuple] = {
            cid: palette[i] for i, cid in enumerate(self._clusters)
        }


    def _label(self, cid: int) -> str:
        name = self._custom_names.get(cid)
        return f"C{cid} · {name}" if name else f"C{cid}"

    def _labels_list(self) -> list[str]:
        return [self._label(cid) for cid in self._clusters]

    def _close(self) -> np.ndarray:
        if self.windows is None:
            raise ValueError("windows обязателен для графиков профилей / возвратов.")
        return self.windows[:, :, self.close_idx].astype(float)

    def _window_returns(self) -> np.ndarray:
        """(close[-1] / close[0]) - 1 для каждого окна."""
        close = self._close()
        first = close[:, 0]
        last = close[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(first != 0, last / first - 1.0, np.nan)
        return ret

    def plot_embeddings_2d(
        self,
        embeddings_2d: Optional[np.ndarray] = None,
        method: str = "umap",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        perplexity: float = 30.0,
        sample_size: Optional[int] = 15_000,
        title: str = "Cluster Embeddings (2D)",
    ) -> plt.Figure:
        if embeddings_2d is not None:
            E2 = np.asarray(embeddings_2d, dtype=float)
        elif self.embeddings is not None:
            E2 = self._project_2d(
                self.embeddings, method, n_neighbors, min_dist, perplexity
            )
        else:
            raise ValueError("Нужны embeddings_2d или self.embeddings.")

        lbl = self.labels
        idx = np.arange(len(lbl))
        if sample_size and len(idx) > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, sample_size, replace=False)

        fig, ax = plt.subplots(figsize=(11, 8))
        _style(fig)
        _panel(ax)

        noise_idx = idx[lbl[idx] == -1]
        if len(noise_idx):
            ax.scatter(
                E2[noise_idx, 0], E2[noise_idx, 1],
                s=5, c=_NOISE_COLOR, alpha=0.25, label="noise", zorder=1,
                rasterized=True,
            )

        for cid in self._clusters:
            ci = idx[lbl[idx] == cid]
            c = self._colors[cid]
            ax.scatter(
                E2[ci, 0], E2[ci, 1],
                s=9, c=[c], alpha=_ALPHA_POINT,
                label=self._label(cid), zorder=2, rasterized=True,
            )
            cx, cy = E2[ci, 0].mean(), E2[ci, 1].mean()
            ax.scatter(
                cx, cy, s=260, c=[c], marker="X",
                edgecolors="white", linewidths=0.8, zorder=5,
            )
            ax.annotate(
                self._label(cid), (cx, cy),
                fontsize=8, color="white", fontweight="bold",
                xytext=(6, 6), textcoords="offset points",
                zorder=6,
            )

        n_noise = (lbl == -1).sum()
        n_valid = self._valid.sum()
        ax.set_title(
            f"{title}  ·  {len(self._clusters)} clusters, "
            f"{n_valid:,} points, {n_noise:,} noise",
            fontsize=_FONT_TITLE, pad=14, color=_TEXT,
        )
        ax.set_xlabel("dim 1", fontsize=_FONT_LABEL)
        ax.set_ylabel("dim 2", fontsize=_FONT_LABEL)
        ax.legend(
            loc="upper right", markerscale=2, fontsize=9,
            framealpha=0.25, edgecolor=_GRID, labelcolor=_TEXT,
            facecolor=_PANEL,
        )
        plt.tight_layout()
        return fig

    def _project_2d(
        self, X: np.ndarray, method: str,
        n_neighbors: int, min_dist: float, perplexity: float,
    ) -> np.ndarray:
        method = method.lower()
        if method == "umap":
            try:
                import umap as _umap
                return _umap.UMAP(
                    n_components=2, n_neighbors=n_neighbors,
                    min_dist=min_dist, random_state=42,
                ).fit_transform(X)
            except ImportError:
                warnings.warn("umap-learn не установлен; используется PCA.")
                method = "pca"
        if method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(
                n_components=2, perplexity=perplexity, random_state=42
            ).fit_transform(X)
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(X)

    def plot_mean_window_profiles(
        self,
        normalize: bool = True,
        max_cols: int = 3,
    ) -> plt.Figure:
        close = self._close()
        n_c = len(self._clusters)
        ncols = min(max_cols, n_c)
        nrows = (n_c + ncols - 1) // ncols
        T = close.shape[1]
        x = np.arange(T)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5.2, nrows * 3.8 + 0.6),
            sharey=False,
        )
        _style(fig)
        axes_flat = np.array(axes).flatten()

        for i, cid in enumerate(self._clusters):
            ax = axes_flat[i]
            _panel(ax)

            mask = self.labels == cid
            C = close[mask].copy()
            if normalize:
                mn = C.min(axis=1, keepdims=True)
                mx = C.max(axis=1, keepdims=True)
                denom = np.where(mx - mn == 0, 1.0, mx - mn)
                C = (C - mn) / denom

            mean = C.mean(axis=0)
            std = C.std(axis=0)
            color = self._colors[cid]

            ax.fill_between(x, mean - std, mean + std, color=color, alpha=_ALPHA_FILL)
            ax.plot(x, mean, color=color, lw=2.5, zorder=3)

            delta = mean[-1] - mean[0]
            if abs(delta) < 0.05:
                arrow = "→"
                arrow_color = "#636c76"
            elif delta > 0:
                arrow = "▲"
                arrow_color = "#1a7f37"
            else:
                arrow = "▼"
                arrow_color = "#cf222e"

            count = int(mask.sum())
            ax.set_title(
                f"{self._label(cid)}  ", fontsize=10, color=_TEXT, loc="left",
            )
            ax.text(
                0.98, 0.95, f"{arrow}  n={count:,}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=arrow_color, fontweight="bold",
            )
            ax.set_xlabel("шаг окна", fontsize=8)
            ax.set_ylabel("norm close" if normalize else "close", fontsize=8)

            ax.axvline(0, color=color, lw=0.8, ls="--", alpha=0.4)
            ax.axvline(T - 1, color=color, lw=0.8, ls="--", alpha=0.4)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            "Средний профиль окна по кластерам  (mean ± std)",
            fontsize=_FONT_TITLE, color=_TEXT, y=1.01,
        )
        plt.tight_layout()
        return fig

    def plot_cluster_stats(self) -> plt.Figure:
        close = self._close()
        returns = self._window_returns()

        stat_keys = ["mean_return", "volatility", "skewness", "size"]
        stat_titles = [
            "Средний Return окна (%)",
            "Волатильность (intra-window, %)",
            "Скewness Returns",
            "Размер кластера (окон)",
        ]
        stats: dict[str, list] = {k: [] for k in stat_keys}

        for cid in self._clusters:
            mask = self.labels == cid
            close_c = close[mask]
            ret_c = returns[mask]
            ret_c = ret_c[~np.isnan(ret_c)]

            with np.errstate(divide="ignore", invalid="ignore"):
                log_c = np.where(close_c > 0, np.log(close_c), np.nan)
                log_ret = np.diff(log_c, axis=1)
            vol = float(np.nanstd(log_ret, axis=1).mean()) * 100

            stats["mean_return"].append(float(np.nanmean(ret_c)) * 100 if len(ret_c) else 0.0)
            stats["volatility"].append(vol)
            stats["skewness"].append(
                float(scipy_stats.skew(ret_c)) if len(ret_c) > 2 else 0.0
            )
            stats["size"].append(int(mask.sum()))

        x_labels = self._labels_list()
        colors = [self._colors[cid] for cid in self._clusters]

        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        _style(fig)

        for ax, key, ttl in zip(axes.flat, stat_keys, stat_titles):
            _panel(ax)
            vals = stats[key]
            bars = ax.bar(
                x_labels, vals, color=colors,
                width=0.55, edgecolor="none", zorder=3,
            )
            ax.set_title(ttl, fontsize=10, pad=8)
            ax.tick_params(axis="x", rotation=25)
            ax.axhline(0, color=_TEXT, lw=0.6, alpha=0.3, zorder=2)

            span = max(abs(v) for v in vals) if vals else 1.0
            for bar, val in zip(bars, vals):
                y = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + span * 0.04 * (1 if y >= 0 else -1),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=8, color=_TEXT,
                )

        fig.suptitle("Статистики кластеров", fontsize=_FONT_TITLE, color=_TEXT)
        plt.tight_layout()
        return fig

    def plot_feature_heatmap(self) -> plt.Figure:
        if self.windows is None:
            raise ValueError("windows обязателен для plot_feature_heatmap.")
        if self.feature_names is None:
            raise ValueError("feature_names обязателен для plot_feature_heatmap.")

        last = self.windows[:, -1, :].astype(float)
        feat_df = pd.DataFrame(last, columns=self.feature_names)
        feat_df["_cluster"] = self.labels
        valid = feat_df[feat_df["_cluster"] != -1]
        medians = valid.groupby("_cluster")[self.feature_names].median()

        med_z = (medians - medians.mean()) / (medians.std() + 1e-8)

        label_map = {cid: self._label(cid) for cid in medians.index}
        med_z_disp = med_z.rename(index=label_map)
        med_disp = medians.rename(index=label_map)

        n_f = len(self.feature_names)
        n_c = len(medians)
        fig, ax = plt.subplots(
            figsize=(max(13, n_f * 0.75), max(4, n_c + 1))
        )
        _style(fig)
        ax.set_facecolor(_PANEL)

        sns.heatmap(
            med_z_disp,
            ax=ax, cmap="RdBu_r", center=0,
            annot=med_disp.round(3), fmt="",
            linewidths=0.5, linecolor=_GRID,
            cbar_kws={"label": "z-score по признаку", "shrink": 0.8},
        )
        ax.set_title(
            "Медианы признаков по кластерам  (аннотация = медиана, цвет = z-score)",
            fontsize=_FONT_TITLE, color=_TEXT, pad=12,
        )
        ax.tick_params(colors=_TEXT, labelsize=8)
        ax.set_ylabel("Кластер", fontsize=_FONT_LABEL, color=_TEXT)
        ax.set_xlabel("Признак", fontsize=_FONT_LABEL, color=_TEXT)
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color=_TEXT, labelsize=8)
            cbar.ax.yaxis.label.set_color(_TEXT)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig

    def plot_returns_distribution(self) -> plt.Figure:
        returns = self._window_returns()

        rows = []
        for cid in self._clusters:
            mask = self.labels == cid
            ret = returns[mask]
            ret = ret[~np.isnan(ret)]
            for r in ret:
                rows.append({"cluster": self._label(cid), "return": r * 100})

        df = pd.DataFrame(rows)
        order = self._labels_list()
        palette = {self._label(cid): self._colors[cid] for cid in self._clusters}

        fig, ax = plt.subplots(
            figsize=(max(9, len(self._clusters) * 1.6 + 2), 6)
        )
        _style(fig)
        _panel(ax)

        sns.violinplot(
            data=df, x="cluster", y="return", order=order,
            palette=palette, inner=None, cut=0.5,
            ax=ax, linewidth=0, alpha=0.55,
        )
        sns.boxplot(
            data=df, x="cluster", y="return", order=order,
            palette=palette, width=0.18, fliersize=2.5,
            linewidth=1.2, ax=ax,
            boxprops=dict(alpha=0.95, zorder=4),
            whiskerprops=dict(color=_TEXT, linewidth=0.8),
            capprops=dict(color=_TEXT, linewidth=0.8),
            medianprops=dict(color=_TEXT, linewidth=1.8),
            flierprops=dict(marker=".", markerfacecolor=_TEXT, markersize=3, alpha=0.4),
        )
        ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(
            "Распределение returns по кластерам",
            fontsize=_FONT_TITLE, pad=12,
        )
        ax.set_xlabel("Кластер", fontsize=_FONT_LABEL)
        ax.set_ylabel("Return окна (%)", fontsize=_FONT_LABEL)
        plt.tight_layout()
        return fig

    def plot_temporal_distribution(
        self,
        selected_tickers: Optional[list[str]] = None,
        max_tickers: int = 8,
        aggregate: bool = False,
    ) -> plt.Figure:
        if self.timestamps is None:
            raise ValueError("timestamps обязателен для temporal plot.")

        if aggregate or self.tickers is None:
            return self._plot_temporal_area()
        return self._plot_temporal_gantt(selected_tickers, max_tickers)

    def _plot_temporal_area(self) -> plt.Figure:
        ts = pd.to_datetime(self.timestamps)
        df = pd.DataFrame({"date": ts, "cluster": self.labels})
        df = df[df["cluster"] != -1].copy()
        df = df.sort_values("date")
        df["month"] = df["date"].dt.to_period("M")

        pivot = (
            df.groupby(["month", "cluster"])
            .size()
            .unstack(fill_value=0)
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
        pivot.index = pivot.index.to_timestamp()

        fig, ax = plt.subplots(figsize=(15, 5))
        _style(fig)
        _panel(ax)

        cols_in_pivot = [c for c in self._clusters if c in pivot.columns]
        colors = [self._colors[c] for c in cols_in_pivot]
        ax.stackplot(
            pivot.index,
            [pivot[c] for c in cols_in_pivot],
            labels=[self._label(c) for c in cols_in_pivot],
            colors=colors, alpha=0.82,
        )
        ax.set_xlim(pivot.index[0], pivot.index[-1])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Доля кластера", fontsize=_FONT_LABEL)
        ax.set_xlabel("Дата", fontsize=_FONT_LABEL)
        ax.set_title(
            "Динамика кластеров во времени", fontsize=_FONT_TITLE, pad=12,
        )
        ax.legend(
            loc="upper left", fontsize=9,
            framealpha=0.25, facecolor=_PANEL, edgecolor=_GRID,
            labelcolor=_TEXT,
        )
        plt.tight_layout()
        return fig

    def _plot_temporal_gantt(
        self,
        selected_tickers: Optional[list[str]],
        max_tickers: int,
    ) -> plt.Figure:
        ts = np.asarray(self.timestamps)
        tickers = self.tickers

        if selected_tickers is None:
            unique, counts = np.unique(tickers, return_counts=True)
            order = np.argsort(-counts)
            selected_tickers = unique[order][:max_tickers].tolist()

        n = len(selected_tickers)
        fig, ax = plt.subplots(figsize=(16, max(3.5, n * 0.65 + 1.8)))
        _style(fig)
        _panel(ax)

        for row_i, ticker in enumerate(selected_tickers):
            mask = tickers == ticker
            t_ts = ts[mask]
            t_labels = self.labels[mask]
            order = np.argsort(t_ts)
            t_ts = t_ts[order]
            t_labels = t_labels[order]

            for j in range(len(t_ts)):
                cid = int(t_labels[j])
                color = _NOISE_COLOR if cid == -1 else self._colors.get(cid, "#888")
                x0 = pd.Timestamp(t_ts[j])
                x1 = (
                    pd.Timestamp(t_ts[j + 1])
                    if j + 1 < len(t_ts)
                    else x0 + pd.Timedelta(days=20)
                )
                dur = (x1 - x0).days
                ax.barh(
                    row_i, dur, left=x0, height=0.72,
                    color=color, alpha=0.88, edgecolor="none",
                )

        ax.set_yticks(range(n))
        ax.set_yticklabels(selected_tickers, fontsize=9, color=_TEXT)
        ax.set_xlabel("Дата", fontsize=_FONT_LABEL, color=_TEXT)
        ax.tick_params(colors=_TEXT)
        ax.set_title(
            "Рыночные режимы по тикерам", fontsize=_FONT_TITLE, color=_TEXT, pad=12,
        )

        patches = [
            mpatches.Patch(color=self._colors[cid], label=self._label(cid))
            for cid in self._clusters
        ]
        ax.legend(
            handles=patches,
            loc="lower right",
            ncol=min(5, len(self._clusters)),
            fontsize=8, framealpha=0.25,
            facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT,
        )
        plt.tight_layout()
        return fig

    def plot_transition_heatmap(self) -> plt.Figure:
        counts = pd.DataFrame(
            0.0, index=self._clusters, columns=self._clusters
        )

        if self.tickers is not None and self.timestamps is not None:
            for ticker in np.unique(self.tickers):
                mask = self.tickers == ticker
                t_ts = self.timestamps[mask]
                t_labels = self.labels[mask]
                order_idx = np.argsort(t_ts)
                seq = t_labels[order_idx]
                for a, b in zip(seq[:-1], seq[1:]):
                    ai, bi = int(a), int(b)
                    if ai in counts.index and bi in counts.columns:
                        counts.loc[ai, bi] += 1
        else:
            seq = self.labels
            for a, b in zip(seq[:-1], seq[1:]):
                ai, bi = int(a), int(b)
                if ai in counts.index and bi in counts.columns:
                    counts.loc[ai, bi] += 1

        row_sums = counts.sum(axis=1).replace(0, np.nan)
        prob = counts.div(row_sums, axis=0).fillna(0.0)

        label_map = {cid: self._label(cid) for cid in self._clusters}
        prob = prob.rename(index=label_map, columns=label_map)

        n = len(prob)
        fig, ax = plt.subplots(figsize=(max(7, n + 3), max(6, n + 2)))
        _style(fig)
        _panel(ax)

        mask_diag = np.eye(n, dtype=bool)

        sns.heatmap(
            prob, ax=ax,
            cmap="YlOrRd",
            annot=True, fmt=".2f",
            linewidths=0.6, linecolor=_GRID,
            vmin=0, vmax=1,
            cbar_kws={"label": "вероятность перехода", "shrink": 0.8},
            mask=mask_diag,
        )
        diag_df = pd.DataFrame(
            np.where(mask_diag, prob.values, np.nan),
            index=prob.index, columns=prob.columns,
        )
        sns.heatmap(
            diag_df, ax=ax,
            cmap=sns.light_palette("#1a7f37", as_cmap=True),
            annot=True, fmt=".2f",
            linewidths=0.6, linecolor=_GRID,
            vmin=0, vmax=1,
            cbar=False,
            mask=~mask_diag,
        )

        ax.set_title(
            "Матрица переходов между кластерами  P[ строка → столбец ]",
            fontsize=_FONT_TITLE, color=_TEXT, pad=14,
        )
        ax.set_xlabel("В кластер", fontsize=_FONT_LABEL, color=_TEXT)
        ax.set_ylabel("Из кластера", fontsize=_FONT_LABEL, color=_TEXT)
        ax.tick_params(colors=_TEXT, labelsize=9)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)

        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color=_TEXT, labelsize=8)
            cbar.ax.yaxis.label.set_color(_TEXT)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT)

        plt.tight_layout()
        return fig

    def cluster_metrics(
        self,
        X: Optional[np.ndarray] = None,
        sample_size: Optional[int] = 50_000,
    ) -> pd.DataFrame:
        if X is not None:
            X_base = np.asarray(X, dtype=float)
        elif self.embeddings is not None:
            X_base = np.asarray(self.embeddings, dtype=float)
        elif self.windows is not None:
            X_base = self.windows.reshape(len(self.windows), -1).astype(float)
        else:
            raise ValueError(
                "Нужен хотя бы один из: X, self.embeddings, self.windows."
            )

        mask = self.labels != -1
        X_valid = X_base[mask]
        y_valid = self.labels[mask]

        n_valid = int(mask.sum())
        n_noise = int((~mask).sum())
        n_clusters = len(self._clusters)

        if n_clusters < 2:
            rows = [
                {"metric": "Silhouette Score",      "value": float("nan"), "note": "< 2 кластеров"},
                {"metric": "Davies-Bouldin Index",  "value": float("nan"), "note": "< 2 кластеров"},
                {"metric": "Calinski-Harabasz Index","value": float("nan"), "note": "< 2 кластеров"},
            ]
            return pd.DataFrame(rows)

        if sample_size and n_valid > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_valid, sample_size, replace=False)
            sil = silhouette_score(X_valid[idx], y_valid[idx])
            sil_note = f"sample {sample_size:,} / {n_valid:,}"
        else:
            sil = silhouette_score(X_valid, y_valid)
            sil_note = f"n={n_valid:,}"

        db = davies_bouldin_score(X_valid, y_valid)
        ch = calinski_harabasz_score(X_valid, y_valid)

        noise_pct = n_noise / len(self.labels) * 100

        rows = [
            {
                "metric": "Silhouette Score",
                "value": round(sil, 4),
                "interpretation": "↑ лучше  (−1 … +1)",
                "note": sil_note,
            },
            {
                "metric": "Davies-Bouldin Index",
                "value": round(db, 4),
                "interpretation": "↓ лучше  (≥ 0)",
                "note": f"n={n_valid:,}",
            },
            {
                "metric": "Calinski-Harabasz Index",
                "value": round(ch, 2),
                "interpretation": "↑ лучше  (≥ 0)",
                "note": f"n={n_valid:,}",
            },
            {
                "metric": "Noise ratio",
                "value": round(noise_pct, 2),
                "interpretation": "↓ лучше  (%)",
                "note": f"{n_noise:,} / {len(self.labels):,} точек",
            },
            {
                "metric": "N clusters",
                "value": float(n_clusters),
                "interpretation": "—",
                "note": f"без шума",
            },
        ]
        return pd.DataFrame(rows).set_index("metric")

    def run_all(
        self,
        embeddings_2d: Optional[np.ndarray] = None,
        X_metrics: Optional[np.ndarray] = None,
        selected_tickers: Optional[list[str]] = None,
        max_tickers: int = 8,
        aggregate_temporal: bool = False,
    ) -> dict[str, Optional[plt.Figure]]:
        figs: dict[str, Optional[plt.Figure]] = {}
        sep = "═" * 58

        print(sep)
        print("  ClusterAnalyzer")
        print(f"  {len(self._clusters)} кластеров · {self._valid.sum():,} валидных окон"
              f" · {(~self._valid).sum():,} шум")
        print(sep)

        try:
            df_metrics = self.cluster_metrics(X=X_metrics)
            figs["metrics"] = df_metrics
            print("\nМетрики качества кластеризации")
            try:
                from IPython.display import display
                display(df_metrics)
            except Exception:
                print(df_metrics.to_string())
        except Exception as exc:
            print(f"  Метрики — ошибка: {exc}")
            figs["metrics"] = None

        if embeddings_2d is not None or self.embeddings is not None:
            print("\n[1/7] 2D scatter эмбеддингов…")
            figs["embeddings_2d"] = self.plot_embeddings_2d(embeddings_2d=embeddings_2d)
        else:
            print("\n[1/7] 2D scatter — пропущен (нет embeddings)")
            figs["embeddings_2d"] = None

        if self.windows is not None:
            print("[2/7] Средние профили окон…")
            figs["mean_profiles"] = self.plot_mean_window_profiles()
            print("[3/7] Статистики кластеров…")
            figs["stats"] = self.plot_cluster_stats()
            if self.feature_names is not None:
                print("[4/7] Тепловая карта признаков…")
                figs["feature_heatmap"] = self.plot_feature_heatmap()
            else:
                print("[4/7] Тепловая карта — пропущена (нет feature_names)")
                figs["feature_heatmap"] = None
            print("[5/7] Распределение returns…")
            figs["returns_dist"] = self.plot_returns_distribution()
        else:
            print("[2-5/7] windows не задан — пропускаем профили / статистики.")
            figs["mean_profiles"] = figs["stats"] = None
            figs["feature_heatmap"] = figs["returns_dist"] = None

        if self.timestamps is not None:
            print("[6/7] Временное распределение…")
            figs["temporal"] = self.plot_temporal_distribution(
                selected_tickers=selected_tickers,
                max_tickers=max_tickers,
                aggregate=aggregate_temporal,
            )
        else:
            print("[6/7] Временное — пропущен (нет timestamps)")
            figs["temporal"] = None

        print("[7/7] Матрица переходов…")
        figs["transition"] = self.plot_transition_heatmap()

        print(f"\n{sep}")
        print("Готово")
        print(sep)
        return figs
