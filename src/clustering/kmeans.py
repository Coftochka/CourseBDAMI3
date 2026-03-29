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
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale = scale
        self.n_init = n_init

        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        X = self._scale_fit(X)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self._kmeans.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._kmeans.predict(self._scale_transform(X))

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
        self._check_fitted()
        return self._kmeans.cluster_centers_

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return X

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            return self._scaler.transform(X)
        return X
