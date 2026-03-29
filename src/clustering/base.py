from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseClusterer(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def labels_(self) -> np.ndarray: ...

    def cluster_sizes(self) -> dict[int, int]:
        labels = self.labels_
        unique, counts = np.unique(labels, return_counts=True)
        return {int(k): int(c) for k, c in zip(unique, counts)}

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseClusterer":
        with open(path, "rb") as f:
            return pickle.load(f)
