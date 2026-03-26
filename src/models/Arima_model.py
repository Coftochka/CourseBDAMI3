from Base_model import BaseModel
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ARIMA работает только в режиме "single" — одна акция.
#
# Таргет y строим заранее (shift, лог-доходность).
# Модель обучается на временном ряду y и предсказывает
# следующее значение (однощаговый прогноз).
#
# Для каждого окна предсказания используется rolling-прогноз:
#   predict(X, y) — для каждой позиции i берём y[:i] и предсказываем y[i]
#
# auto_order=True автоматически подбирает порядок (p, d, q)
# через перебор по AIC; иначе используются переданные p, d, q.


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
        """
        p          : AR order
        d          : differencing order
        q          : MA order
        auto_order : if True, grid-searches (p, d, q) by AIC on fit data
        p_max      : max AR order for grid search
        q_max      : max MA order for grid search
        trend      : trend term — "n" (none) | "c" (const) | "t" | "ct"
        """
        self.p = p
        self.d = d
        self.q = q
        self.auto_order = auto_order
        self.p_max = p_max
        self.q_max = q_max
        self.trend = trend

        self._order: Tuple[int, int, int] = (p, d, q)
        self._fitted_model = None
        self._train_y: Optional[np.ndarray] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Stationarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_d(y: np.ndarray, max_d: int = 2, significance: float = 0.05) -> int:
        """ADF test to suggest differencing order."""
        for d in range(max_d + 1):
            series = np.diff(y, n=d) if d > 0 else y
            pval = adfuller(series, autolag="AIC")[1]
            if pval < significance:
                return d
        return max_d

    def _grid_search(self, y: np.ndarray) -> Tuple[int, int, int]:
        """Find best (p, d, q) by AIC on the training series."""
        d = self._suggest_d(y)
        best_aic = np.inf
        best_order = (1, d, 1)

        for p in range(self.p_max + 1):
            for q in range(self.q_max + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    res = ARIMA(y, order=(p, d, q), trend=self.trend).fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

        print(f"[ArimaModel] best order={best_order}  AIC={best_aic:.2f}")
        return best_order

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y: np.ndarray,
        X_val=None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        X   : ignored (kept for API compatibility)
        y   : (n_timesteps,) target time series (e.g. log-returns)

        Fits ARIMA on the full training series y.
        If auto_order=True, runs grid search first.
        """
        y = np.asarray(y, dtype=np.float64)

        if self.auto_order:
            if verbose:
                print("[ArimaModel] Running grid search for (p, d, q)...")
            self._order = self._grid_search(y)
        else:
            self._order = (self.p, self.d, self.q)

        self._train_y = y

        if verbose:
            print(f"[ArimaModel] Fitting ARIMA{self._order} on {len(y)} samples...")

        result = ARIMA(y, order=self._order, trend=self.trend).fit()
        self._fitted_model = result

        train_pred = result.fittedvalues
        train_mse = float(mean_squared_error(y, train_pred))
        self.history["train_loss"] = [train_mse]

        if verbose:
            print(f"[ArimaModel] Train MSE={train_mse:.6f}  AIC={result.aic:.2f}  BIC={result.bic:.2f}")

        if y_val is not None:
            y_val = np.asarray(y_val, dtype=np.float64)
            val_pred = self._rolling_predict(y, y_val)
            val_mse = float(mean_squared_error(y_val, val_pred))
            self.history["val_loss"] = [val_mse]
            if verbose:
                print(f"[ArimaModel] Val   MSE={val_mse:.6f}")

    # ------------------------------------------------------------------
    # Rolling predict
    # ------------------------------------------------------------------

    def _rolling_predict(self, history: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        """
        One-step-ahead rolling forecast on y_new,
        using history as the initial training window.
        """
        assert self._fitted_model is not None, "call fit() first"
        preds = []
        current_history = list(history)

        for val in y_new:
            res = ARIMA(current_history, order=self._order, trend=self.trend).fit()
            forecast = float(res.forecast(steps=1).iloc[0])
            preds.append(forecast)
            current_history.append(val)

        return np.array(preds, dtype=np.float32)

    def predict(self, X, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Rolling one-step-ahead forecast over y.

        X   : ignored
        y   : (n_timesteps,) — the target series to forecast over.
              Each prediction at position i uses y[:train_len + i] as history.

        Returns:
            pred : (n_timesteps,) float
        """
        assert self._fitted_model is not None, "call fit() first"
        assert y is not None, "y (target series) is required for ARIMA predict()"

        y = np.asarray(y, dtype=np.float64)
        assert self._train_y is not None
        return self._rolling_predict(self._train_y, y)

    def predict_last(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Forecast `steps` steps ahead from the end of y.

        y     : (n_timesteps,) — history to condition on
        steps : number of steps to forecast

        Returns:
            forecast : (steps,) float
        """
        assert self._fitted_model is not None, "call fit() first"
        y = np.asarray(y, dtype=np.float64)
        res = ARIMA(y, order=self._order, trend=self.trend).fit()
        return res.forecast(steps=steps).to_numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _regression_metrics_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> pd.DataFrame:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])

        return pd.DataFrame({
            "mse":          mse,
            "rmse":         np.sqrt(mse),
            "mae":          mae,
            "r2":           r2,
            "dir_accuracy": dir_acc,
            "ic":           ic,
        }, index=[model_name])

    def scores(
        self,
        X,
        y: np.ndarray,
        model_name: str = "ARIMA",
    ) -> pd.DataFrame:
        """Regression metrics on rolling one-step predictions over y."""
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X, y)
        return self._regression_metrics_df(y, y_pred, model_name)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_loss(self):
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])
        if not train:
            raise RuntimeError("fit model before plotting loss")

        labels, values = ["Train MSE"], [train[0]]
        if val:
            labels.append("Val MSE")
            values.append(val[0])

        plt.figure(figsize=(5, 4))
        bars = plt.bar(labels, values, color=["steelblue", "salmon"][:len(values)])
        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{v:.6f}", ha="center", va="bottom", fontsize=9)
        plt.ylabel("MSE")
        plt.title(f"ARIMA{self._order} loss")
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self):
        """Residual diagnostics plot from statsmodels."""
        assert self._fitted_model is not None, "call fit() first"
        self._fitted_model.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print statsmodels ARIMA summary."""
        assert self._fitted_model is not None, "call fit() first"
        print(self._fitted_model.summary())

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Saves the fitted model and config to a pickle file.
        path : file path (e.g. "arima.pkl")
        """
        payload = {
            "config": {
                "p":          self.p,
                "d":          self.d,
                "q":          self.q,
                "auto_order": self.auto_order,
                "p_max":      self.p_max,
                "q_max":      self.q_max,
                "trend":      self.trend,
            },
            "order":        self._order,
            "train_y":      self._train_y,
            "fitted_model": self._fitted_model,
            "history":      self.history,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "ArimaModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(**payload["config"])
        obj._order = payload["order"]
        obj._train_y = payload["train_y"]
        obj._fitted_model = payload["fitted_model"]
        obj.history = payload["history"]
        return obj
