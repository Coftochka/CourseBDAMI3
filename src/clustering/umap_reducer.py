"""
UMAP для снижения размерности эмбеддингов перед кластеризацией.

Типичный пайплайн:
    proj = UMAPProjector(n_components=15, n_neighbors=30)
    Z_train = proj.fit_transform(emb_train)
    Z_val   = proj.transform(emb_val)

    clusterer = EmbeddingClusterer(n_clusters=5, scale=False)  # уже в «умап-пространстве»
    clusterer.fit(Z_train)

Почему UMAP перед кластеризацией:
    • убирает «шумовые» измерения и кривизну многообразия;
    • кластеры в 2D–20D часто разделяются лучше, чем в исходных 64D;
    • для HDBSCAN / KMeans важно, чтобы расстояния были более «локальными».

scale=True (по умолчанию): StandardScaler fit только на train — как и для кластеризаторов.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


class UMAPProjector:
    """
    Обёртка над umap-learn: fit на train, transform на новых точках.

    Параметры UMAP подобраны как разумные стартовые для табличных эмбеддингов;
    при необходимости смотрите документацию umap-learn для тюнинга.
    """

    def __init__(
        self,
        n_components: int = 15,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        scale: bool = True,
        low_memory: bool = False,
    ):
        """
        n_components : целевая размерность (обычно 10–50 перед кластеризацией).
        n_neighbors  : локальность vs глобальная структура (больше → глобальнее).
        min_dist       : насколько плотно «сжимать» точки (меньше → плотнее кластеры).
        metric         : расстояние в исходном пространстве эмбеддингов.
        scale          : StandardScaler по признакам (fit только на train).
        low_memory     : передать в UMAP для больших данных.
        """
        self.n_components = n_components
        self.n_neighbors  = n_neighbors
        self.min_dist     = min_dist
        self.metric       = metric
        self.random_state = random_state
        self.scale        = scale
        self.low_memory   = low_memory

        self._scaler: Optional[StandardScaler] = None
        self._umap = None

    def _build_umap(self):
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "Нужен пакет umap-learn. Установите: pip install umap-learn"
            ) from e

        return umap.UMAP(
            n_components   = self.n_components,
            n_neighbors    = self.n_neighbors,
            min_dist       = self.min_dist,
            metric         = self.metric,
            random_state   = self.random_state,
            low_memory     = self.low_memory,
            verbose        = False,
        )

    def fit(self, X: np.ndarray) -> "UMAPProjector":
        """
        X : (N, D) — эмбеддинги train.
        Обучает scaler (если нужен) и UMAP.
        """
        X = np.asarray(X, dtype=np.float32)
        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._umap = self._build_umap()
        self._umap.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X : (M, D) — те же D, что при fit.
        Returns : (M, n_components)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return np.asarray(self._umap.transform(X), dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Одна выборка: обучение + проекция за один проход (umap.fit_transform)."""
        X = np.asarray(X, dtype=np.float32)
        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._umap = self._build_umap()
        return np.asarray(self._umap.fit_transform(X), dtype=np.float32)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "UMAPProjector":
        with open(path, "rb") as f:
            return pickle.load(f)

    def _check_fitted(self):
        if self._umap is None:
            raise RuntimeError("Сначала вызовите fit() или fit_transform().")
