"""
Cluster quality analysis and visualisation utilities.

    from src.clustering.cluster_analysis import plot_elbow, plot_clusters_2d, cluster_stats
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── k selection ───────────────────────────────────────────────────────────────

def elbow_scores(
    embeddings: np.ndarray,
    k_range: range | list[int] = range(2, 11),
    scale: bool = True,
    random_state: int = 42,
) -> dict[str, list]:
    """
    Returns inertia and silhouette scores for each k.
    Use to pick optimal number of clusters before running the full experiment.
    """
    X = StandardScaler().fit_transform(embeddings) if scale else embeddings
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        silhouettes.append(sil)

    return {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}


def plot_elbow(
    embeddings: np.ndarray,
    k_range: range | list[int] = range(2, 11),
    scale: bool = True,
    random_state: int = 42,
    figsize: tuple = (12, 4),
):
    scores = elbow_scores(embeddings, k_range, scale, random_state)
    ks = scores["k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(ks, scores["inertia"], marker="o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow curve")
    ax1.grid(alpha=0.3)

    ax2.plot(ks, scores["silhouette"], marker="o", color="orange")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette score")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return scores


# ── 2-D visualisation ─────────────────────────────────────────────────────────

def plot_clusters_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scale: bool = True,
    figsize: tuple = (8, 6),
    title: str = "Clusters (PCA 2D)",
    alpha: float = 0.4,
    s: int = 8,
):
    """Project embeddings to 2D with PCA and colour by cluster label."""
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


# ── per-cluster statistics ────────────────────────────────────────────────────

def cluster_stats(
    labels: np.ndarray,
    returns: np.ndarray | None = None,
) -> "pd.DataFrame":
    """
    labels  : (N,) cluster assignments
    returns : (N,) optional array of target values (e.g. log-returns)

    Returns a DataFrame with per-cluster counts and, if returns provided,
    mean/std/median of returns.
    """
    import pandas as pd

    ks = np.unique(labels)
    rows = []
    for k in ks:
        mask = labels == k
        row: dict = {"cluster": int(k), "count": int(mask.sum())}
        if returns is not None:
            r = returns[mask]
            row["return_mean"]   = float(r.mean())
            row["return_std"]    = float(r.std())
            row["return_median"] = float(np.median(r))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("cluster")
    df["pct"] = df["count"] / df["count"].sum() * 100
    return df
