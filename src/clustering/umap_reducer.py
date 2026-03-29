from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import umap
from sklearn.preprocessing import StandardScaler


class UMAPProjector:

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
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.scale = scale
        self.low_memory = low_memory

        self._scaler: Optional[StandardScaler] = None
        self._umap = None

    def _build_umap(self):
        return umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            low_memory=self.low_memory,
            verbose=False,
        )

    def fit(self, X: np.ndarray) -> "UMAPProjector":
        X = np.asarray(X, dtype=np.float32)
        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._umap = self._build_umap()
        self._umap.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return np.asarray(self._umap.transform(X), dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
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
