"""
KMeans кластеризатор эмбеддингов.

    clusterer = KMeansClusterer(n_clusters=5)
    clusterer.fit(emb_train)
    labels_val = clusterer.predict(emb_val)
    clusterer.save("src/data/clusters/kmeans_k5.pkl")
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import BaseClusterer


class KMeansClusterer(BaseClusterer):

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        scale: bool = True,
        n_init: int = 10,
    ):
        """
        n_clusters   : число кластеров
        scale        : StandardScaler (fit только на train)
        n_init       : число запусков с разными центроидами
        """
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.scale        = scale
        self.n_init       = n_init

        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans]         = None
        self._fitted = False

    # ── fit / predict ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        X = self._scale_fit(X)
        self._kmeans = KMeans(
            n_clusters   = self.n_clusters,
            random_state = self.random_state,
            n_init       = self.n_init,
        )
        self._kmeans.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._kmeans.predict(self._scale_transform(X))

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def labels_(self) -> np.ndarray:
        self._check_fitted()
        return self._kmeans.labels_

    @property
    def inertia_(self) -> float:
        self._check_fitted()
        return self._kmeans.inertia_

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Центроиды в (возможно масштабированном) пространстве."""
        self._check_fitted()
        return self._kmeans.cluster_centers_

    # ── internal ──────────────────────────────────────────────────────────────

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return X

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            return self._scaler.transform(X)
        return X
