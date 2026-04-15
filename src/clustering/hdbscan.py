from __future__ import annotations

from typing import Any, Optional

import numpy as np
from hdbscan import HDBSCAN as _HDBSCAN
from hdbscan import approximate_predict, membership_vector
from sklearn.preprocessing import StandardScaler

from .base import BaseClusterer


class HDBSCANClusterer(BaseClusterer):
    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        max_cluster_size: int = 0,
        metric: str = "euclidean",
        alpha: float = 1.0,
        p: Optional[float] = None,
        algorithm: str = "best",
        leaf_size: int = 40,
        approx_min_span_tree: bool = True,
        gen_min_span_tree: bool = False,
        core_dist_n_jobs: int = 4,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        prediction_data: bool = True,
        branch_detection_data: bool = False,
        match_reference_implementation: bool = False,
        cluster_selection_epsilon_max: float = float("inf"),
        scale: bool = True,
        **kwargs: Any,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.prediction_data = prediction_data
        self.branch_detection_data = branch_detection_data
        self.match_reference_implementation = match_reference_implementation
        self.cluster_selection_epsilon_max = cluster_selection_epsilon_max
        self.scale = scale
        self.kwargs = kwargs

        self._scaler: Optional[StandardScaler] = None
        self._hdbscan: Optional[_HDBSCAN] = None


    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        X = self._scale_fit(X)
        self._hdbscan = _HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_persistence=self.cluster_selection_persistence,
            max_cluster_size=self.max_cluster_size,
            metric=self.metric,
            alpha=self.alpha,
            p=self.p,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            approx_min_span_tree=self.approx_min_span_tree,
            gen_min_span_tree=self.gen_min_span_tree,
            core_dist_n_jobs=self.core_dist_n_jobs,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            prediction_data=self.prediction_data,
            branch_detection_data=self.branch_detection_data,
            match_reference_implementation=self.match_reference_implementation,
            cluster_selection_epsilon_max=self.cluster_selection_epsilon_max,
            **self.kwargs,
        )
        self._hdbscan.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._scale_transform(X)
        labels, _ = approximate_predict(self._hdbscan, X)
        return np.asarray(labels, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._scale_transform(X)
        return np.asarray(membership_vector(self._hdbscan, X))


    @property
    def labels_(self) -> np.ndarray:
        return np.asarray(self._hdbscan.labels_)

    @property
    def probabilities_(self) -> np.ndarray:
        return np.asarray(self._hdbscan.probabilities_)

    @property
    def n_clusters_(self) -> int:
        return int(self.labels_.max()) + 1

    @property
    def noise_ratio_(self) -> float:
        labels = self.labels_
        return float((labels == -1).sum() / len(labels))

    @property
    def outlier_scores_(self) -> np.ndarray:    
        return np.asarray(self._hdbscan.outlier_scores_)

    @property
    def minimum_spanning_tree_(self):
        return self._hdbscan.minimum_spanning_tree_

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return X

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            return np.asarray(self._scaler.transform(X))
        return X
