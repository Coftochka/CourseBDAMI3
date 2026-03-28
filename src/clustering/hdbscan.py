"""
HDBSCAN кластеризатор эмбеддингов.

Использует пакет `hdbscan` (не sklearn), поскольку только он предоставляет
`approximate_predict` — настоящее предсказание кластеров для новых точек
без переобучения модели.

    clusterer = HDBSCANClusterer(min_cluster_size=50)
    clusterer.fit(emb_train)

    print(clusterer.n_clusters_)   # найденное число кластеров (без шума)
    print(clusterer.noise_ratio_)  # доля шумовых точек (label == -1)

    labels_val = clusterer.predict(emb_val)  # approximate_predict
    clusterer.save("src/data/clusters/hdbscan.pkl")

Установка: pip install hdbscan
"""
from __future__ import annotations

from typing import Optional

import numpy as np
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
        """
        min_cluster_size          : минимальный размер кластера — главный параметр.
        min_samples               : плотность ядра; None → равно min_cluster_size.
        cluster_selection_epsilon : порог слияния (0 = отключён).
        metric                    : расстояние ("euclidean", "cosine", …).
        scale                     : StandardScaler перед кластеризацией.
        prediction_data           : строить prediction_data для approximate_predict
                                    (небольшие доп. расходы памяти/времени при fit).
        """
        self.min_cluster_size          = min_cluster_size
        self.min_samples               = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric                    = metric
        self.scale                     = scale
        self.prediction_data           = prediction_data

        self._scaler: Optional[StandardScaler] = None
        self._hdbscan: Optional[object] = None
        self._fitted = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        try:
            import hdbscan as _hdbscan_pkg
        except ImportError as e:
            raise ImportError("Установите пакет: pip install hdbscan") from e

        X = self._scale_fit(X)
        self._hdbscan = _hdbscan_pkg.HDBSCAN(
            min_cluster_size          = self.min_cluster_size,
            min_samples               = self.min_samples,
            cluster_selection_epsilon = self.cluster_selection_epsilon,
            metric                    = self.metric,
            prediction_data           = self.prediction_data,
        )
        self._hdbscan.fit(X)
        self._fitted = True
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Назначает метки новым точкам через `hdbscan.approximate_predict`.
        Шумовые точки получают метку -1.
        """
        self._check_fitted()
        from hdbscan import approximate_predict

        X = self._scale_transform(X)
        result = approximate_predict(self._hdbscan, X)
        labels = result[0]
        return np.asarray(labels, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности принадлежности к кластеру для каждой точки.
        Строки соответствуют точкам, столбцы — кластерам (по возрастанию id).
        Шумовые точки имеют вероятность 0 по всем кластерам.
        """
        self._check_fitted()
        from hdbscan import membership_vector

        X = self._scale_transform(X)
        return np.asarray(membership_vector(self._hdbscan, X))

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def labels_(self) -> np.ndarray:
        """Метки, назначенные fit()-данным. -1 = шум."""
        self._check_fitted()
        return np.asarray(getattr(self._hdbscan, "labels_"))

    @property
    def probabilities_(self) -> np.ndarray:
        """Вероятность принадлежности к кластеру (0 для шумовых точек)."""
        self._check_fitted()
        return np.asarray(getattr(self._hdbscan, "probabilities_"))

    @property
    def n_clusters_(self) -> int:
        """Число найденных кластеров (без учёта шума)."""
        self._check_fitted()
        return int(self.labels_.max()) + 1

    @property
    def noise_ratio_(self) -> float:
        """Доля точек, отнесённых к шуму (label == -1)."""
        self._check_fitted()
        labels = self.labels_
        return float((labels == -1).sum() / len(labels))

    # ── internal ──────────────────────────────────────────────────────────────

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return X

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            return np.asarray(self._scaler.transform(X))
        return X
