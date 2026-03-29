from .Base_model import BaseModel
from typing import Optional
import numpy as np
import lightgbm as lgb


class LightGBMModel(BaseModel):
    """
    LightGBM regression model for windowed time-series.

    Internally flattens X from (N, seq_len, num_features) → (N, seq_len * num_features).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self._model: Optional[lgb.LGBMRegressor] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """(N, seq_len, F) → (N, seq_len*F).  Already-2D arrays pass through."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def _make_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        X_flat = self._flatten(X)
        y = np.asarray(y, dtype=np.float32)

        self._model = self._make_model()

        fit_params: dict = {}
        eval_sets = [(X_flat, y)]
        eval_names = ["train"]

        if X_val is not None and y_val is not None:
            X_val_flat = self._flatten(X_val)
            y_val = np.asarray(y_val, dtype=np.float32)
            eval_sets.append((X_val_flat, y_val))
            eval_names.append("valid")
            fit_params["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        fit_params["eval_set"] = eval_sets
        fit_params["eval_names"] = eval_names

        self._model.fit(X_flat, y, **fit_params)

        # Store loss history
        res = self._model.evals_result_
        self.train_losses_: list[float] = list(res.get("train", {}).get("l2", []))
        self.val_losses_: list[float] = list(res.get("valid", {}).get("l2", []))

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() first"
        X_flat = self._flatten(X)
        return self._model.predict(X_flat).astype(np.float32)

    def plot_loss(self, title: str = "") -> None:
        """Plot LightGBM train/val loss curves."""
        import matplotlib.pyplot as plt

        if not getattr(self, "train_losses_", None):
            return
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.train_losses_, label="Train loss")
        if self.val_losses_:
            ax.plot(self.val_losses_, label="Val loss")
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel("L2 (MSE)")
        ax.set_title(title or "LightGBM training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
