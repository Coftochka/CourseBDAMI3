from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


def classification_metrics_df(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    return pd.DataFrame(metrics, index=[model_name])


class BaseModel(ABC):
    
    @abstractmethod
    def fit(self, X, y, **kwargs): ...
    
    @abstractmethod
    def predict(self, X): ...
    
    @abstractmethod
    def save(self, path: str): ...
    
    @abstractmethod
    def load(self, path: str): ...