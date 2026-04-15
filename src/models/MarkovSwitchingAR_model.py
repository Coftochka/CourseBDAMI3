import math
import warnings
from typing import Optional

import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from .Base_model import BaseModel


class MarkovSwitchingARModel(BaseModel):

    def __init__(
        self,
        k_regimes: int = 2,
        order: int = 1,
        switching_variance: bool = True,
    ):
        self.k_regimes = k_regimes
        self.order = order
        self.switching_variance = switching_variance
        self._fitted = False

    @staticmethod
    def _extract_series(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X[:, :, 0]
        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        series_all = self._extract_series(X)

        n_transition = self.k_regimes * (self.k_regimes - 1)
        block_size = 1 + self.order + (1 if self.switching_variance else 0)

        preds: list[float] = []
        for window in series_all:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = MarkovAutoregression(
                        window,
                        k_regimes=self.k_regimes,
                        order=self.order,
                        switching_variance=self.switching_variance,
                    )
                    res = mod.fit(disp=False, maxiter=100)

                last_probs = np.array(res.smoothed_marginal_probabilities[-1])
                params = res.params
                y_last = float(window[-1])

                y_hat = 0.0
                for k in range(self.k_regimes):
                    offset = n_transition + k * block_size
                    const_k = float(params[offset])
                    ar_term = sum(
                        float(params[offset + 1 + j]) * float(window[-(j + 1)])
                        for j in range(self.order)
                    )
                    y_hat += float(last_probs[k]) * (const_k + ar_term)

                log_ret = y_hat - y_last
                if math.isnan(log_ret) or math.isinf(log_ret):
                    log_ret = 0.0
                preds.append(log_ret)

            except Exception:
                preds.append(0.0)

        result = np.array(preds, dtype=np.float32)
        bad = ~np.isfinite(result)
        if bad.any():
            result[bad] = 0.0
        return result
