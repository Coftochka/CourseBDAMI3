from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def _kmeans_metrics(
    X: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[float, float, float, float]:
    """KMeans + все метрики для заданного k."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    if k > 1:
        sil = float(silhouette_score(X, labels))
        ch  = float(calinski_harabasz_score(X, labels))
        db  = float(davies_bouldin_score(X, labels))
    else:
        sil = ch = db = float("nan")
    return float(km.inertia_), sil, ch, db


def elbow_scores(
    embeddings: np.ndarray,
    k_range: range | list[int] = range(2, 11),
    scale: bool = True,
    random_state: int = 42,
    n_jobs: int | None = -1,
) -> dict[str, list]:
    """
    Возвращает inertia, silhouette, calinski_harabasz, davies_bouldin для каждого k.

    Parameters
    ----------
    n_jobs : int or None
        ``-1`` — все CPU; ``1`` / ``None`` — последовательно.
    """
    X = StandardScaler().fit_transform(embeddings) if scale else np.asarray(
        embeddings, dtype=np.float64
    )
    ks = list(k_range)
    if not ks:
        return {"k": [], "inertia": [], "silhouette": [], "calinski_harabasz": [], "davies_bouldin": []}

    if n_jobs is None or n_jobs == 1:
        inertias, sils, chs, dbs = [], [], [], []
        for k in ks:
            inertia, sil, ch, db = _kmeans_metrics(X, k, random_state)
            inertias.append(inertia); sils.append(sil)
            chs.append(ch); dbs.append(db)
    else:
        workers = os.cpu_count() or 1
        workers = min(workers, abs(n_jobs) if n_jobs > 0 else workers, len(ks))
        workers = max(1, workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda k: _kmeans_metrics(X, k, random_state), ks))
        inertias = [r[0] for r in results]
        sils     = [r[1] for r in results]
        chs      = [r[2] for r in results]
        dbs      = [r[3] for r in results]

    return {"k": ks, "inertia": inertias, "silhouette": sils,
            "calinski_harabasz": chs, "davies_bouldin": dbs}


def _elbow_k_index(ks: list | np.ndarray, inertias: list | np.ndarray) -> int:
    """
    Индекс k с «локтем»: максимальное расстояние точки (k, inertia) от хорды
    первой и последней точки в нормированных координатах.
    """
    k_arr = np.asarray(ks, dtype=float)
    y_arr = np.asarray(inertias, dtype=float)
    n = len(k_arr)
    if n < 2:
        return 0
    k_n = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-12)
    y_n = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min() + 1e-12)
    x0, y0 = k_n[0], y_n[0]
    x1, y1 = k_n[-1], y_n[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.hypot(dx, dy) + 1e-12
    dists = np.abs((k_n - x0) * dy - (y_n - y0) * dx) / denom
    return int(np.argmax(dists))


def plot_elbow(
    embeddings: np.ndarray,
    k_range: range | list[int] = range(2, 11),
    scale: bool = True,
    random_state: int = 42,
    figsize: tuple = (16, 4),
    n_jobs: int | None = -1,
):
    """
    Строит три графика: Elbow (inertia), Silhouette, Calinski-Harabasz.

    Почему три метрики:
    - Silhouette bias к k=2 на непрерывных распределениях (финансовые данные).
    - Calinski-Harabasz = between/within variance ratio, менее bias.
    - Elbow (inertia) — геометрический ориентир.
    Принимай k по согласию хотя бы двух метрик.
    """
    scores = elbow_scores(embeddings, k_range, scale, random_state, n_jobs=n_jobs)
    ks       = scores["k"]
    inertias = scores["inertia"]
    sils     = np.asarray(scores["silhouette"],       dtype=float)
    chs      = np.asarray(scores["calinski_harabasz"], dtype=float)

    i_elbow = _elbow_k_index(ks, inertias)
    k_elbow = ks[i_elbow]
    scores["k_elbow"] = k_elbow

    i_sil = int(np.nanargmax(sils)) if np.any(np.isfinite(sils)) else None
    k_sil = ks[i_sil] if i_sil is not None else None
    scores["k_silhouette_max"] = k_sil

    i_ch  = int(np.nanargmax(chs)) if np.any(np.isfinite(chs)) else None
    k_ch  = ks[i_ch] if i_ch is not None else None
    scores["k_calinski_max"] = k_ch

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # ── Elbow ────────────────────────────────────────────────────────────────
    ax1.plot(ks, inertias, marker="o", color="C0")
    ax1.scatter([k_elbow], [inertias[i_elbow]], s=180, c="crimson", marker="*",
                zorder=3, edgecolors="darkred", lw=1.2, label=f"elbow k={k_elbow}")
    ax1.axvline(k_elbow, color="crimson", ls="--", alpha=0.35)
    ax1.set(xlabel="k", ylabel="Inertia (WCSS)", title="Elbow curve")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9)

    # ── Silhouette ───────────────────────────────────────────────────────────
    ax2.plot(ks, sils, marker="o", color="C1")
    if i_sil is not None:
        ax2.scatter([k_sil], [sils[i_sil]], s=180, c="forestgreen", marker="*",
                    zorder=3, edgecolors="darkgreen", lw=1.2,
                    label=f"max k={k_sil} ({sils[i_sil]:.3f})")
        ax2.axvline(k_sil, color="forestgreen", ls="--", alpha=0.35)
        ax2.legend(fontsize=9)
    ax2.set(xlabel="k", ylabel="Silhouette", title="Silhouette  (bias→k=2)")
    ax2.grid(alpha=0.3)

    # ── Calinski-Harabasz ────────────────────────────────────────────────────
    ax3.plot(ks, chs, marker="o", color="C2")
    if i_ch is not None:
        ax3.scatter([k_ch], [chs[i_ch]], s=180, c="darkorange", marker="*",
                    zorder=3, edgecolors="saddlebrown", lw=1.2,
                    label=f"max k={k_ch} ({chs[i_ch]:.0f})")
        ax3.axvline(k_ch, color="darkorange", ls="--", alpha=0.35)
        ax3.legend(fontsize=9)
    ax3.set(xlabel="k", ylabel="CH score", title="Calinski-Harabasz  (↑ лучше)")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return scores


def plot_clusters_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scale: bool = True,
    figsize: tuple = (8, 6),
    title: str = "Clusters (PCA 2D)",
    alpha: float = 0.4,
    s: int = 8,
):
    X = StandardScaler().fit_transform(embeddings) if scale else embeddings
    coords = PCA(n_components=2).fit_transform(X)

    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", n_clusters)

    plt.figure(figsize=figsize)
    for k in range(n_clusters):
        mask = labels == k
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    color=cmap(k), label=f"Cluster {k}",
                    alpha=alpha, s=s)
    plt.legend(markerscale=3)
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.show()


def cluster_stats(
    labels: np.ndarray,
    returns: np.ndarray | None = None,
) -> pd.DataFrame:
    ks = np.unique(labels)
    rows = []
    for k in ks:
        mask = labels == k
        row: dict = {"cluster": int(k), "count": int(mask.sum())}
        if returns is not None:
            r = returns[mask]
            row["return_mean"] = float(r.mean())
            row["return_std"] = float(r.std())
            row["return_median"] = float(np.median(r))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("cluster")
    df["pct"] = df["count"] / df["count"].sum() * 100
    return df
