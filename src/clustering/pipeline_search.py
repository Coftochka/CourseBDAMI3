from __future__ import annotations

import time
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder

try:
    import hdbscan as _hdbscan_pkg
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan не установлен — HDBSCAN недоступен.")

try:
    import umap as _umap_pkg
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    warnings.warn("umap-learn не установлен.")


def _kneedle_elbow(cumvar: np.ndarray, max_k: int = 200) -> int:
    n = min(len(cumvar), max_k)
    x = np.linspace(0.0, 1.0, n)
    y = (cumvar[:n] - cumvar[0]) / max(cumvar[n - 1] - cumvar[0], 1e-9)
    distances = np.abs(y - x)
    return int(np.argmax(distances)) + 1


def pca_analysis(
    X: np.ndarray,
    random_state: int = 42,
    plot: bool = True,
) -> dict[str, Any]:
    pca = PCA(random_state=random_state)
    pca.fit(X)
    ev = pca.explained_variance_ratio_
    cumvar = np.cumsum(ev)

    elbow_k = _kneedle_elbow(cumvar)

    thresholds = {t: int(np.searchsorted(cumvar, t) + 1) for t in (0.80, 0.90, 0.95)}

    if plot:
        n_show = min(60, len(ev))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].bar(range(1, n_show + 1), ev[:n_show],
                    color="steelblue", alpha=0.75, width=0.8)
        axes[0].axvline(thresholds[0.90], color="crimson", ls="--", lw=1.5,
                        label=f"90% → {thresholds[0.90]} компонент")
        axes[0].axvline(elbow_k, color="darkorange", ls=":", lw=1.8,
                        label=f"elbow → {elbow_k}")
        axes[0].set_title(f"Scree plot (первые {n_show} компонент)")
        axes[0].set_xlabel("Компонента")
        axes[0].set_ylabel("Объяснённая дисперсия")
        axes[0].legend(fontsize=9)

        axes[1].plot(range(1, len(cumvar) + 1), cumvar, lw=2, color="steelblue")
        colors_ = {0.80: "royalblue", 0.90: "crimson", 0.95: "darkorange"}
        for t, color in colors_.items():
            k = thresholds[t]
            axes[1].axhline(t, color=color, ls=":", lw=1.2, alpha=0.7)
            axes[1].axvline(k, color=color, ls=":", lw=1.2, alpha=0.7)
            axes[1].annotate(
                f"{int(t * 100)}% → {k}", xy=(k, t),
                xytext=(k + 3, t - 0.07), fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )
        axes[1].axvline(elbow_k, color="darkorange", ls=":", lw=1.8)
        axes[1].set_xlim(0, min(len(cumvar), 100))
        axes[1].set_xlabel("Число компонент")
        axes[1].set_ylabel("Накопленная дисперсия")
        axes[1].set_title("Cumulative explained variance")

        plt.suptitle(
            f"PCA: {X.shape[1]}D вход  |  "
            f"Elbow: {elbow_k}  |  90%: {thresholds[0.90]}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()

    print("PCA рекомендации:")
    print(f"  Elbow (kneedle): {elbow_k}")
    for t, k in thresholds.items():
        print(f"  {int(t * 100)}% дисперсии → {k} компонент")

    return {
        "pca": pca,
        "cumvar": cumvar,
        "elbow_k": elbow_k,
        **{f"k{int(t * 100)}": k for t, k in thresholds.items()},
    }


class ClusterPipelineSearch:
    _STABILITY_SEEDS = [42, 0, 99]
    _VALID_ALGOS = {"hdbscan", "kmeans"}

    def __init__(
        self,
        *,
        n_pca_values: list[int],
        n_neighbors_values: list[int] = (10, 20, 30, 50),
        min_dist_values: list[float] = (0.0, 0.1),
        n_components_umap_values: list[int] = (5, 10, 15),
        algos: list[str] = ("hdbscan", "kmeans"),
        min_cluster_size_values: list[int] = (50, 100, 200, 500),
        n_clusters_values: list[int] = (3, 4, 5, 6, 8),
        min_n_clusters: int = 3,
        max_n_clusters: int = 8,
        max_noise_ratio: float = 0.20,
        min_stability_ari: float = 0.70,
        min_cluster_points: int = 300,
        w_silhouette: float = 0.4,
        w_downstream: float = 0.4,
        w_stability: float = 0.2,
        sil_sample: int = 10_000,
        n_jobs: int = -1,
        save_path: Optional[str] = None,
        random_state: int = 42,
    ):
        unknown = set(algos) - self._VALID_ALGOS
        if unknown:
            raise ValueError(f"Неизвестные алгоритмы: {unknown}. Допустимы: {self._VALID_ALGOS}")
        self.algos = list(algos)

        self.n_pca_values = list(n_pca_values)
        self.n_neighbors_values = list(n_neighbors_values)
        self.min_dist_values = list(min_dist_values)
        self.n_components_umap_values = list(n_components_umap_values)
        self.min_cluster_size_values = list(min_cluster_size_values)
        self.n_clusters_values = list(n_clusters_values)

        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.max_noise_ratio = max_noise_ratio
        self.min_stability_ari = min_stability_ari
        self.min_cluster_points = min_cluster_points

        self.w_silhouette = w_silhouette
        self.w_downstream = w_downstream
        self.w_stability = w_stability
        self.sil_sample = sil_sample
        self.n_jobs = n_jobs
        self.save_path = Path(save_path) if save_path else None
        self.random_state = random_state

        self.results_: Optional[pd.DataFrame] = None
        self._rows: list[dict] = []


    def run(self, X_scaled: np.ndarray, y_target: np.ndarray) -> "ClusterPipelineSearch":
        X = np.asarray(X_scaled, dtype=float)
        y = np.asarray(y_target, dtype=float)

        baseline_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))

        umap_combos = list(product(
            self.n_pca_values,
            self.n_neighbors_values,
            self.min_dist_values,
            self.n_components_umap_values,
        ))
        run_hdbscan = "hdbscan" in self.algos and _HDBSCAN_AVAILABLE
        run_kmeans  = "kmeans"  in self.algos
        n_hdb = len(self.min_cluster_size_values) if run_hdbscan else 0
        n_km  = len(self.n_clusters_values)       if run_kmeans  else 0
        total = len(umap_combos) * (n_hdb + n_km)

        algo_info = "  +  ".join(
            ([f"{n_hdb} HDBSCAN"] if run_hdbscan else [])
            + ([f"{n_km} KMeans"]  if run_kmeans  else [])
        )
        print(f"Алгоритмы: {self.algos}")
        print(f"UMAP-групп: {len(umap_combos)}  ×  ({algo_info}) = {total} конфигураций")
        print(f"Каждая UMAP-группа запускается 3× для stability. "
              f"≈ {len(umap_combos) * 3} UMAP fits.\n")

        pbar = self._tqdm(umap_combos, desc="UMAP groups", unit="group")
        for n_pca, n_neighbors, min_dist, n_umap in pbar:
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    n_pca=n_pca, nn=n_neighbors, md=min_dist, nu=n_umap
                )

            try:
                pca = PCA(n_components=min(n_pca, X.shape[1] - 1),
                          random_state=self.random_state)
                P = pca.fit_transform(X)
            except Exception as exc:
                warnings.warn(f"PCA(n={n_pca}) failed: {exc}")
                continue

            try:
                Z_main = self._fit_umap(
                    P, n_umap, n_neighbors, min_dist,
                    seed=self.random_state, n_jobs=1,
                )
            except Exception as exc:
                warnings.warn(f"UMAP(main) failed: {exc}")
                continue

            Z_list: list[Optional[np.ndarray]] = [Z_main]
            for _ in range(len(self._STABILITY_SEEDS) - 1):
                try:
                    Z_list.append(self._fit_umap(
                        P, n_umap, n_neighbors, min_dist,
                        seed=None, n_jobs=self.n_jobs,   # параллельный
                    ))
                except Exception as exc:
                    warnings.warn(f"UMAP(stability) failed: {exc}")
                    Z_list.append(None)

            umap_key = dict(
                n_pca=n_pca, n_neighbors=n_neighbors,
                min_dist=min_dist, n_components_umap=n_umap,
            )

            if run_hdbscan:
                for mcs in self.min_cluster_size_values:
                    self._eval(
                        Z_main, Z_list, y, baseline_rmse,
                        algo="hdbscan",
                        algo_params={"min_cluster_size": mcs, "n_clusters": None},
                        umap_key=umap_key,
                    )

            if run_kmeans:
                for k in self.n_clusters_values:
                    self._eval(
                        Z_main, Z_list, y, baseline_rmse,
                        algo="kmeans",
                        algo_params={"min_cluster_size": None, "n_clusters": k},
                        umap_key=umap_key,
                    )

            self._checkpoint()

        self.results_ = pd.DataFrame(self._rows) if self._rows else pd.DataFrame()
        self._add_composite_score()
        if self.save_path and len(self.results_):
            self.results_.to_csv(self.save_path, index=False)
            print(f"\nРезультаты сохранены: {self.save_path}")
        return self


    def filtered(self) -> pd.DataFrame:
        df = self.results_
        if df is None or len(df) == 0:
            return pd.DataFrame()

        m = (
            (df["n_clusters"] >= self.min_n_clusters)
            & (df["n_clusters"] <= self.max_n_clusters)
            & (df["stability_ari"].fillna(0) >= self.min_stability_ari)
            & (df["min_cluster_size_actual"].fillna(0) >= self.min_cluster_points)
        )
        hdb = df["algo"] == "hdbscan"
        m = m & (~hdb | (df["noise_ratio"].fillna(1.0) <= self.max_noise_ratio))

        return df[m].sort_values("composite_score", ascending=False)

    def top_k(self, k: int = 3, filtered: bool = True) -> pd.DataFrame:
        cols = [
            "rank", "algo",
            "n_pca", "n_neighbors", "min_dist", "n_components_umap",
            "min_cluster_size", "n_clusters", "noise_ratio",
            "silhouette", "davies_bouldin",
            "stability_ari", "downstream_rmse", "downstream_improvement",
            "composite_score",
        ]
        df = self.filtered() if filtered else self.results_.sort_values(
            "composite_score", ascending=False
        )
        if len(df) == 0 and filtered:
            print("Нет конфигураций, прошедших фильтры. Показываем топ без фильтрации.")
            df = self.results_.sort_values("composite_score", ascending=False)

        top = df.head(k).reset_index(drop=True)
        top.insert(0, "rank", range(1, len(top) + 1))
        present = [c for c in cols if c in top.columns]
        return top[present]

    def summary(self) -> None:
        df = self.results_
        if df is None or len(df) == 0:
            print("Нет результатов.")
            return
        f = self.filtered()
        print(f"Всего конфигураций: {len(df)}")
        print(f"Прошли жёсткие фильтры: {len(f)}")
        if len(df):
            print(f"\nSilhouette — min: {df['silhouette'].min():.4f}  "
                  f"max: {df['silhouette'].max():.4f}  "
                  f"mean: {df['silhouette'].mean():.4f}")
            print(f"Stability ARI — min: {df['stability_ari'].min():.4f}  "
                  f"max: {df['stability_ari'].max():.4f}")
            print(f"Noise ratio (HDBSCAN) — "
                  f"min: {df.loc[df['algo']=='hdbscan','noise_ratio'].min():.3f}  "
                  f"max: {df.loc[df['algo']=='hdbscan','noise_ratio'].max():.3f}")


    @staticmethod
    def _fit_umap(
        P: np.ndarray,
        n_components: int,
        n_neighbors: int,
        min_dist: float,
        seed: Optional[int],
        n_jobs: int = 1,
    ) -> np.ndarray:
        if not _UMAP_AVAILABLE:
            raise ImportError("umap-learn не установлен.")
        reducer = _umap_pkg.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=seed,
            n_jobs=n_jobs,
            low_memory=True,
            verbose=False,
        )
        return reducer.fit_transform(P)

    def _cluster(
        self, Z: np.ndarray, algo: str, params: dict
    ) -> np.ndarray:
        if algo == "hdbscan":
            h = _hdbscan_pkg.HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                core_dist_n_jobs=4,
                prediction_data=False,
            )
            return np.asarray(h.fit_predict(Z), dtype=int)
        elif algo == "kmeans":
            km = KMeans(
                n_clusters=params["n_clusters"],
                random_state=self.random_state,
                n_init=10,
            )
            return np.asarray(km.fit_predict(Z), dtype=int)
        raise ValueError(f"Неизвестный алгоритм: {algo}")

    def _stability_ari(
        self,
        Z_list: list[Optional[np.ndarray]],
        algo: str,
        params: dict,
    ) -> float:
        label_sets: list[np.ndarray] = []
        for Z in Z_list:
            if Z is None:
                continue
            try:
                label_sets.append(self._cluster(Z, algo, params))
            except Exception:
                pass
        if len(label_sets) < 2:
            return float("nan")
        aris = [
            adjusted_rand_score(label_sets[i], label_sets[j])
            for i, j in combinations(range(len(label_sets)), 2)
        ]
        return float(np.mean(aris))

    def _downstream_rmse(self, labels: np.ndarray, y: np.ndarray) -> float:
        """RMSE Ridge-регрессии cluster_label → y_target (one-hot features)."""
        mask = labels != -1
        if mask.sum() == 0:
            return float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        lbl_v, y_v = labels[mask], y[mask]
        if len(np.unique(lbl_v)) < 2:
            return float(np.sqrt(np.mean((y_v - y_v.mean()) ** 2)))
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_enc = enc.fit_transform(lbl_v.reshape(-1, 1))
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_enc, y_v)
        return float(np.sqrt(mean_squared_error(y_v, ridge.predict(X_enc))))

    def _eval(
        self,
        Z_main: np.ndarray,
        Z_list: list[Optional[np.ndarray]],
        y: np.ndarray,
        baseline_rmse: float,
        algo: str,
        algo_params: dict,
        umap_key: dict,
    ) -> None:
        t0 = time.time()
        try:
            lbl = self._cluster(Z_main, algo, algo_params)
            n_c = int(len(np.unique(lbl[lbl != -1])))
            noise_ratio = float((lbl == -1).sum() / len(lbl))
            mask = lbl != -1

            sil, db = float("nan"), float("nan")
            if n_c >= 2 and mask.sum() > 100:
                n_samp = min(self.sil_sample, int(mask.sum()))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sil = float(silhouette_score(
                        Z_main[mask], lbl[mask],
                        sample_size=n_samp, random_state=self.random_state,
                    ))
                    db = float(davies_bouldin_score(Z_main[mask], lbl[mask]))

            stab = self._stability_ari(Z_list, algo, algo_params)

            valid_labels = lbl[mask]
            if len(valid_labels):
                _, cnts = np.unique(valid_labels, return_counts=True)
                min_c_size = int(cnts.min())
            else:
                min_c_size = 0

            ds_rmse = self._downstream_rmse(lbl, y)
            ds_impr = max(0.0, (baseline_rmse - ds_rmse) / (baseline_rmse + 1e-9))

            self._rows.append({
                **umap_key,
                "algo":                    algo,
                "min_cluster_size":        algo_params.get("min_cluster_size"),
                "n_clusters":              n_c,
                "noise_ratio":             round(noise_ratio, 4),
                "min_cluster_size_actual": min_c_size,
                "silhouette":              round(sil, 4) if np.isfinite(sil) else float("nan"),
                "davies_bouldin":          round(db, 4) if np.isfinite(db) else float("nan"),
                "stability_ari":           round(stab, 4) if np.isfinite(stab) else float("nan"),
                "downstream_rmse":         round(ds_rmse, 6),
                "downstream_improvement":  round(ds_impr, 4),
                "time_s":                  round(time.time() - t0, 1),
            })
        except Exception as exc:
            warnings.warn(f"{algo} {algo_params} eval failed: {exc}")

    def _add_composite_score(self) -> None:
        df = self.results_
        if df is None or len(df) == 0:
            return

        sil_norm = ((df["silhouette"].fillna(-1) + 1) / 2).clip(0, 1)

        ds = df["downstream_improvement"].fillna(0).clip(0)
        ds_max = ds.max()
        ds_norm = (ds / ds_max).clip(0, 1) if ds_max > 0 else ds

        stab = df["stability_ari"].fillna(0).clip(0, 1)

        df["composite_score"] = (
            self.w_silhouette * sil_norm
            + self.w_downstream * ds_norm
            + self.w_stability * stab
        ).round(4)

    def _checkpoint(self) -> None:
        if not self._rows:
            return
        self.results_ = pd.DataFrame(self._rows)
        self._add_composite_score()
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_.to_csv(self.save_path, index=False)

    @staticmethod
    def _tqdm(iterable, **kwargs):
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, **kwargs)
        except ImportError:
            return iterable
