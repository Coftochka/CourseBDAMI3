from __future__ import annotations

from typing import Any, Optional

import numpy as np
from hdbscan import HDBSCAN as _HDBSCAN
from hdbscan import approximate_predict, membership_vector
from sklearn.preprocessing import StandardScaler

from .base import BaseClusterer


class HDBSCANClusterer(BaseClusterer):
    """
    Тонкая обёртка над ``hdbscan.HDBSCAN``, совместимая с ``BaseClusterer``.

    Поддерживает все параметры оригинальной библиотеки.
    Дополнительный параметр ``scale`` включает StandardScaler перед fit/predict.

    Parameters
    ----------
    min_cluster_size : int
        Минимальный размер кластера. Основной параметр чувствительности.
    min_samples : int or None
        Число соседей для расчёта плотности ядра. None → равно min_cluster_size.
    cluster_selection_epsilon : float
        Минимальное расстояние между кластерами (упрощает мелкую кластеризацию).
    cluster_selection_persistence : float
        Минимальная «жизнь» кластера в дереве. Фильтрует нестабильные кластеры.
    max_cluster_size : int
        Ограничение сверху на размер кластера (0 = без ограничений).
    metric : str
        Метрика расстояния ('euclidean', 'cosine', 'manhattan', …).
    alpha : float
        Коэффициент для mutual reachability distance. Обычно 1.0.
    p : float or None
        Степень для метрики Minkowski (только при metric='minkowski').
    algorithm : {'best', 'generic', 'prims_kdtree', 'prims_balltree', 'boruvka_kdtree', 'boruvka_balltree'}
        Алгоритм построения минимального остовного дерева.
    leaf_size : int
        Размер листа для KD/Ball-tree.
    approx_min_span_tree : bool
        Использовать приближённое MST (быстрее, чуть менее точно).
    gen_min_span_tree : bool
        Сохранить MST в атрибуте minimum_spanning_tree_.
    core_dist_n_jobs : int
        Число параллельных потоков для расчёта расстояний ядра.
    cluster_selection_method : {'eom', 'leaf'}
        'eom' — Excess of Mass (по умолчанию, лучше для переменной плотности).
        'leaf' — выбирает листья дерева (много мелких равных кластеров).
    allow_single_cluster : bool
        Разрешить результат из одного кластера.
    prediction_data : bool
        Вычислить доп. структуры для approximate_predict / membership_vector.
    branch_detection_data : bool
        Данные для обнаружения ветвей внутри кластеров.
    match_reference_implementation : bool
        Строгое соответствие эталонной реализации (медленнее).
    cluster_selection_epsilon_max : float
        Верхняя граница epsilon при cluster_selection_method='eom'.
    scale : bool
        Применять StandardScaler до fit/predict (удобно, если данные не нормированы).
    **kwargs
        Дополнительные аргументы, передаваемые напрямую в hdbscan.HDBSCAN
        (например, metric_params для кастомных метрик).
    """

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

    # ── fit / predict ──────────────────────────────────────────────────────────

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
        """Мягкое предсказание через approximate_predict (требует prediction_data=True)."""
        X = self._scale_transform(X)
        labels, _ = approximate_predict(self._hdbscan, X)
        return np.asarray(labels, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности принадлежности к кластерам (требует prediction_data=True)."""
        X = self._scale_transform(X)
        return np.asarray(membership_vector(self._hdbscan, X))

    # ── properties ─────────────────────────────────────────────────────────────

    @property
    def labels_(self) -> np.ndarray:
        return np.asarray(self._hdbscan.labels_)

    @property
    def probabilities_(self) -> np.ndarray:
        """Мягкие вероятности принадлежности для обучающих точек."""
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
        """GLOSH-оценка выбросов (чем выше — тем более выброс)."""
        return np.asarray(self._hdbscan.outlier_scores_)

    @property
    def minimum_spanning_tree_(self):
        """MST как объект hdbscan (только при gen_min_span_tree=True)."""
        return self._hdbscan.minimum_spanning_tree_

    # ── scaling helpers ────────────────────────────────────────────────────────

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return X

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            return np.asarray(self._scaler.transform(X))
        return X
