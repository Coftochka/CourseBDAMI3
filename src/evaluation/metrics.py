"""
Regression metrics for time-series forecasting evaluation.

All functions accept plain numpy arrays and return plain Python floats
so they work with any model (LSTM, RNN, ARIMA, …).

    from src.evaluation.metrics import regression_metrics

    metrics = regression_metrics(y_true, y_pred, model_name="LSTM")
    # → pd.DataFrame with one row
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where predicted sign matches true sign."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between true and predicted returns."""
    if np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    cluster: int | str | None = None,
) -> pd.DataFrame:
    """
    Compute a standard set of regression metrics.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays of the same length
    model_name     : row label in the result DataFrame
    cluster        : optional cluster index — added as a column if provided

    Returns
    -------
    pd.DataFrame with one row
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)

    row: dict = {
        "mae":          mean_absolute_error(y_true, y_pred),
        "rmse":         float(np.sqrt(mse)),
        "mse":          float(mse),
        "mape":         mape(y_true, y_pred),
        "r2":           r2_score(y_true, y_pred),
        "dir_accuracy": directional_accuracy(y_true, y_pred),
        "ic":           information_coefficient(y_true, y_pred),
        "n_samples":    len(y_true),
    }

    if cluster is not None:
        row["cluster"] = cluster

    return pd.DataFrame(row, index=[model_name])
