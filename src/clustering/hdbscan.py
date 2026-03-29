from __future__ import annotations

from typing import Optional

import numpy as np
import hdbscan as _hdbscan_pkg
from hdbscan import approximate_predict, membership_vector
from sklearn.preprocessing import StandardScaler

from .base import BaseClusterer


class HDBSCANClusterer(BaseClusterer):

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        scale: bool = True,
        prediction_data: bool = True,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.scale = scale
        self.prediction_data = prediction_data

        self._scaler: Optional[StandardScaler] = None
        self._hdbscan: Optional[object] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        X = self._scale_fit(X)
        self._hdbscan = _hdbscan_pkg.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            prediction_data=self.prediction_data,
        )
        self._hdbscan.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._scale_transform(X)
        labels = approximate_predict(self._hdbscan, X)[0]
        return np.asarray(labels, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._scale_transform(X)
        return np.asarray(membership_vector(self._hdbscan, X))

    @property
    def labels_(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._hdbscan.labels_)

    @property
    def probabilities_(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._hdbscan.probabilities_)

    @property
    def n_clusters_(self) -> int:
        self._check_fitted()
        return int(self.labels_.max()) + 1

    @property
    def noise_ratio_(self) -> float:
        self._check_fitted()
        labels = self.labels_
        return float((labels == -1).sum() / len(labels))

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return X

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            return np.asarray(self._scaler.transform(X))
        return X
