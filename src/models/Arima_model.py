import warnings
from typing import Optional, Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .Base_model import BaseModel


class ArimaModel(BaseModel):

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        auto_order: bool = False,
        p_max: int = 4,
        q_max: int = 4,
        trend: str = "n",
    ):
        self.p = p
        self.d = d
        self.q = q
        self.auto_order = auto_order
        self.p_max = p_max
        self.q_max = q_max
        self.trend = trend

        self._order: Tuple[int, int, int] = (p, d, q)
        self._fitted = False


    @staticmethod
    def _extract_series(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X[:, :, 0]
        return X

    def _grid_search(self, series: np.ndarray) -> Tuple[int, int, int]:
        best_aic = np.inf
        best_order = (1, self.d, 1)
        for p in range(self.p_max + 1):
            for q in range(self.q_max + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = ARIMA(series, order=(p, self.d, q), trend=self.trend).fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, self.d, q)
                except Exception:
                    continue
        return best_order


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        series_all = self._extract_series(X)

        if self.auto_order:
            sample = series_all[0]
            self._order = self._grid_search(sample)
        else:
            self._order = (self.p, self.d, self.q)

        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        series_all = self._extract_series(X)

        preds: list[float] = []
        for i in range(len(series_all)):
            window = series_all[i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(window, order=self._order, trend=self.trend).fit()
                    fc = model.forecast(steps=1)
                    preds.append(float(fc.iloc[0]))
            except Exception:
                preds.append(float(window[-1]))
        return np.array(preds, dtype=np.float32)
