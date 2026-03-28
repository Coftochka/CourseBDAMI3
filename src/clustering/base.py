"""
Базовый интерфейс кластеризаторов эмбеддингов.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseClusterer(ABC):
    """
    Единый интерфейс для всех кластеризаторов.

    Все реализации обязаны:
        fit(X)        — обучиться на train-эмбеддингах
        predict(X)    — предсказать метки для новых точек
        labels_       — метки, назначенные во время fit
        cluster_sizes — размер каждого кластера
        save / load   — сериализация через pickle

    Шумовые точки (если алгоритм поддерживает) получают метку -1.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """X : (N, D) — эмбеддинги train."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """X : (M, D) → (M,) int — метки кластеров."""

    @property
    @abstractmethod
    def labels_(self) -> np.ndarray:
        """Метки, назначенные во время fit."""

    def cluster_sizes(self) -> dict[int, int]:
        """Размер каждого кластера. Перегрузите при необходимости."""
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

    def _check_fitted(self, attr: str = "_fitted"):
        if not getattr(self, attr, False):
            raise RuntimeError("Call fit() first.")
