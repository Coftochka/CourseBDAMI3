from Base_model import BaseModel
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import json
import copy
import pandas as pd
from typing import Optional, Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Режимы работы модели:
#
#   "single"   — одна акция, ticker_ids не нужны.
#
#   "pooled"   — все акции сразу, одна общая модель.
#                ticker_id добавляется как последняя фича в окно.
#                ticker_ids передаются в fit/predict.
#
#   "finetune" — двухэтапное обучение:
#                1. pretrain(X, y, ticker_ids) — обучение pooled-модели на всех акциях
#                2. finetune(X, y, ticker_id)  — дообучение на одной акции:
#                   копируем pretrained booster и добавляем деревья поверх него
#                   (init_model + n_finetune_estimators с меньшим lr)
#
# Таргет y строим заранее (shift, лог-доходность).
# Для окна X[i : i+seq_len] метка — y[i + seq_len - 1] (последний бар окна).


class LightGBMModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        seq_len: int = 30,
        mode: str = "single",
    ):
        """
        input_size        : number of features per timestep
        n_estimators      : number of boosting rounds
        learning_rate     : shrinkage rate
        num_leaves        : max leaves per tree
        max_depth         : max tree depth (-1 = unlimited)
        min_child_samples : min samples in a leaf
        subsample         : row subsampling ratio per tree
        colsample_bytree  : feature subsampling ratio per tree
        reg_alpha         : L1 regularisation
        reg_lambda        : L2 regularisation
        seq_len           : length of the input sequence window
        mode              : "single" | "pooled" | "finetune"
        """
        assert mode in ("single", "pooled", "finetune"), (
            f"mode is {mode}, should be 'single', 'pooled' or 'finetune'"
        )

        self.input_size = input_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seq_len = seq_len
        self.mode = mode

        self._model: Optional[lgb.LGBMRegressor] = None  # pretrained / single / pooled
        self._finetune_model: Optional[lgb.LGBMRegressor] = None  # finetune copy
        self.feature_names: List[str] = []
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ticker_ids: Optional[np.ndarray] = None,
    ):
        """
        X          : (n_timesteps, n_features)
        y          : (n_timesteps,)
        ticker_ids : (n_timesteps,) int — optional; appended as last feature

        Returns:
        X_win  : (n_windows, seq_len * n_features [+ 1])  — flattened
        y_win  : (n_windows,)
        """
        X_win, y_win = [], []
        max_i = len(X) - self.seq_len + 1
        for i in range(max_i):
            window = X[i: i + self.seq_len].flatten()
            if ticker_ids is not None:
                window = np.append(window, ticker_ids[i + self.seq_len - 1])
            X_win.append(window)
            y_win.append(y[i + self.seq_len - 1])

        return np.array(X_win, dtype=np.float32), np.array(y_win, dtype=np.float32)

    def _build_feature_names(self, n_features: int, with_ticker: bool = False) -> List[str]:
        names = []
        for t in range(self.seq_len):
            lag = self.seq_len - 1 - t
            for f in range(n_features):
                names.append(f"feat_{f}_t-{lag}")
        if with_ticker:
            names.append("ticker_id")
        return names

    def _to_df(self, X_w: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X_w, columns=self.feature_names)

    def _make_lgbm(self, learning_rate: Optional[float] = None) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=learning_rate or self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="regression",
            metric="l2",
            verbose=-1,
        )

    def _fit_lgbm(
        self,
        model: lgb.LGBMRegressor,
        X_w: np.ndarray,
        y_w: np.ndarray,
        X_val_w: Optional[np.ndarray],
        y_val_w: Optional[np.ndarray],
        verbose: bool,
        init_model=None,
    ) -> lgb.LGBMRegressor:
        has_val = X_val_w is not None and y_val_w is not None

        eval_sets  = [(self._to_df(X_w), y_w)]
        eval_names = ["training"]
        if has_val:
            eval_sets.append((self._to_df(X_val_w), y_val_w))
            eval_names.append("valid_0")

        evals_record: dict = {}
        callbacks = [lgb.record_evaluation(evals_record)]
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))

        fit_kwargs = dict(
            X=self._to_df(X_w),
            y=y_w,
            eval_set=eval_sets,
            eval_names=eval_names,
            feature_name=self.feature_names,
            callbacks=callbacks,
        )
        if init_model is not None:
            fit_kwargs["init_model"] = init_model

        model.fit(**fit_kwargs)

        self.history = {"train_loss": [], "val_loss": []}
        self.history["train_loss"] = evals_record.get("training", {}).get("l2", [])
        if has_val:
            self.history["val_loss"] = evals_record.get("valid_0", {}).get("l2", [])

        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        ticker_ids=None,
        X_val=None,
        y_val=None,
        ticker_ids_val=None,
        epochs: int = None,
        verbose: bool = True,
    ):
        """
        X, y         : training data. y — target (float), preprocessed.
        ticker_ids   : (n_timesteps,) int — required for pooled / finetune.

        mode="single":
            fit(X, y, X_val=..., y_val=...)

        mode="pooled":
            fit(X, y, ticker_ids=ids, X_val=..., y_val=..., ticker_ids_val=ids_val)

        mode="finetune":
            Use pretrain() and finetune() instead of fit() directly.
        """
        if self.mode != "single":
            assert ticker_ids is not None, f"ticker_ids is required for mode='{self.mode}'"

        if epochs is not None:
            self.n_estimators = epochs

        with_ticker = ticker_ids is not None
        X_w, y_w = self._make_windows(X, y, ticker_ids)
        self.feature_names = self._build_feature_names(X.shape[1], with_ticker=with_ticker)

        X_val_w, y_val_w = None, None
        if X_val is not None and y_val is not None:
            X_val_w, y_val_w = self._make_windows(X_val, y_val, ticker_ids_val)

        self._model = self._fit_lgbm(
            self._make_lgbm(), X_w, y_w, X_val_w, y_val_w, verbose
        )

    def pretrain(self, X, y, ticker_ids, X_val=None, y_val=None, ticker_ids_val=None, **fit_kwargs):
        """
        Stage 1: train pooled model on all assets.
        Only for mode="finetune".
        """
        assert self.mode == "finetune", "pretrain() is only available for mode='finetune'"
        print("=== Pretrain (all assets) ===")
        self.fit(
            X, y,
            ticker_ids=ticker_ids,
            X_val=X_val, y_val=y_val, ticker_ids_val=ticker_ids_val,
            **fit_kwargs,
        )

    def finetune(
        self,
        X,
        y,
        ticker_id: int,
        X_val=None,
        y_val=None,
        n_finetune_estimators: int = 50,
        finetune_lr: float = 0.01,
        verbose: bool = True,
    ):
        """
        Stage 2: fine-tune on a single asset.
        Only for mode="finetune", after pretrain().

        Copies the pretrained booster and adds n_finetune_estimators trees
        on top of it with a smaller learning rate (init_model trick).

        ticker_id             : index of the asset
        n_finetune_estimators : additional trees to grow
        finetune_lr           : learning rate for the extra trees
        """
        assert self.mode == "finetune", "finetune() is only available for mode='finetune'"
        assert self._model is not None, "call pretrain() before finetune()"
        print(f"=== Finetune (ticker_id={ticker_id}, n_trees={n_finetune_estimators}, lr={finetune_lr}) ===")

        ids = np.full(len(X), ticker_id, dtype=np.int64)
        ids_val = np.full(len(X_val), ticker_id, dtype=np.int64) if X_val is not None else None

        X_w, y_w = self._make_windows(X, y, ids)
        X_val_w, y_val_w = None, None
        if X_val is not None and y_val is not None:
            X_val_w, y_val_w = self._make_windows(X_val, y_val, ids_val)

        # copy pretrained booster — pretrain stays intact
        pretrained_booster = copy.deepcopy(self._model.booster_)

        ft_model = self._make_lgbm(learning_rate=finetune_lr)
        ft_model.n_estimators = n_finetune_estimators

        self._finetune_model = self._fit_lgbm(
            ft_model, X_w, y_w, X_val_w, y_val_w, verbose,
            init_model=pretrained_booster,
        )

    def _active_model(self, finetune: bool = False) -> lgb.LGBMRegressor:
        """Returns finetune model if requested, otherwise the base model."""
        if finetune:
            assert self._finetune_model is not None, "call finetune() first"
            return self._finetune_model
        assert self._model is not None, "model is not fitted yet, call fit() / pretrain() first"
        return self._model

    def predict(self, X, ticker_ids=None, use_finetune: bool = False) -> np.ndarray:
        """
        X              : (n_timesteps, n_features)
        ticker_ids     : (n_timesteps,) int — required for pooled / finetune
        use_finetune   : if True, uses the fine-tuned model (mode="finetune" only)

        Returns:
            pred : (n_windows,) float
        """
        model = self._active_model(finetune=use_finetune)
        X_w, _ = self._make_windows(X, np.zeros(len(X)), ticker_ids)
        return model.predict(self._to_df(X_w)).astype(np.float32)

    def predict_last(self, X, ticker_id: Optional[int] = None, use_finetune: bool = False) -> float:
        """
        X            : (n_timesteps, n_features) — at least seq_len rows
        ticker_id    : int — required for pooled / finetune
        use_finetune : if True, uses the fine-tuned model (mode="finetune" only)

        Returns:
            predicted return : float
        """
        model = self._active_model(finetune=use_finetune)
        if len(X) < self.seq_len:
            raise ValueError(f"need at least {self.seq_len} candles, got {len(X)}")

        window = X[-self.seq_len:].flatten()
        if ticker_id is not None:
            window = np.append(window, ticker_id)
        window = window.reshape(1, -1).astype(np.float32)
        return float(model.predict(self._to_df(window))[0])

    @staticmethod
    def _regression_metrics_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> pd.DataFrame:
        """
        y_true : real returns
        y_pred : predicted returns
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])

        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "dir_accuracy": dir_acc,
            "ic": ic,
        }
        return pd.DataFrame(metrics, index=[model_name])

    def scores(
        self,
        X,
        y,
        model_name: str,
        ticker_ids=None,
        use_finetune: bool = False,
    ) -> pd.DataFrame:
        """
        Regression metrics on the sample.

        For multiple assets, pass ticker_ids — you'll get aggregated metrics.
        For metrics per asset separately, use scores_per_ticker().
        """
        _, y_w = self._make_windows(X, y, ticker_ids)
        y_pred = self.predict(X, ticker_ids, use_finetune=use_finetune)
        return self._regression_metrics_df(y_w, y_pred, model_name)

    def scores_per_ticker(
        self,
        X_dict: Dict[str, np.ndarray],
        y_dict: Dict[str, np.ndarray],
        ticker_to_id: Dict[str, int],
        use_finetune: bool = False,
    ) -> pd.DataFrame:
        """
        X_dict       : {"AAPL": X_aapl, "MSFT": X_msft, ...}
        y_dict       : {"AAPL": y_aapl, "MSFT": y_msft, ...}
        ticker_to_id : {"AAPL": 0, "MSFT": 1, ...}

        Returns:
            DataFrame — one row per ticker + "ALL" (aggregate) row
        """
        frames = []
        all_y, all_pred = [], []

        for ticker, X in X_dict.items():
            y = y_dict[ticker]
            ticker_ids = (
                None if self.mode == "single"
                else np.full(len(X), ticker_to_id[ticker], dtype=np.int64)
            )

            _, y_w = self._make_windows(X, y, ticker_ids)
            y_pred = self.predict(X, ticker_ids, use_finetune=use_finetune)

            frames.append(self._regression_metrics_df(y_w, y_pred, ticker))
            all_y.append(y_w)
            all_pred.append(y_pred)

        frames.append(self._regression_metrics_df(
            np.concatenate(all_y),
            np.concatenate(all_pred),
            "ALL",
        ))
        return pd.concat(frames)

    def plot_loss(self):
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])

        if not train:
            raise RuntimeError("fit model before plotting loss")

        iters = range(1, len(train) + 1)
        plt.figure(figsize=(9, 4))
        plt.plot(iters, train, label="Train loss (MSE)")
        if val:
            plt.plot(iters, val, label="Val loss (MSE)")
        plt.xlabel("Iteration")
        plt.ylabel("MSE (L2)")
        plt.title("Loss by iteration")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_importance(self, use_finetune: bool = False, max_features: int = 20):
        model = self._active_model(finetune=use_finetune)
        lgb.plot_importance(
            model,
            importance_type="gain",
            max_num_features=max_features,
            figsize=(9, 6),
            title="Feature importance (gain)",
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, save_finetune: bool = False):
        """
        Saves the base model (or finetune model if save_finetune=True).

        path            : base path without extension
        save_finetune   : if True, saves _finetune_model instead of _model
        """
        model = self._active_model(finetune=save_finetune)
        suffix = "_finetune" if save_finetune else ""

        model.booster_.save_model(path + suffix + ".txt")
        meta = {
            "config": {
                "input_size":        self.input_size,
                "n_estimators":      self.n_estimators,
                "learning_rate":     self.learning_rate,
                "num_leaves":        self.num_leaves,
                "max_depth":         self.max_depth,
                "min_child_samples": self.min_child_samples,
                "subsample":         self.subsample,
                "colsample_bytree":  self.colsample_bytree,
                "reg_alpha":         self.reg_alpha,
                "reg_lambda":        self.reg_lambda,
                "seq_len":           self.seq_len,
                "mode":              self.mode,
            },
            "feature_names": self.feature_names,
            "is_finetune": save_finetune,
        }
        with open(path + suffix + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, load_as_finetune: bool = False) -> "LightGBMModel":
        """
        path            : base path without extension (and without _finetune suffix)
        load_as_finetune: if True, loads from path_finetune.txt/.json into _finetune_model
        """
        suffix = "_finetune" if load_as_finetune else ""

        with open(path + suffix + ".json") as f:
            meta = json.load(f)

        obj = cls(**meta["config"])
        obj.feature_names = meta["feature_names"]

        booster = lgb.Booster(model_file=path + suffix + ".txt")
        wrapper = lgb.LGBMRegressor()
        wrapper._Booster = booster
        wrapper.fitted_ = True
        wrapper.objective_ = "regression"

        if load_as_finetune:
            obj._finetune_model = wrapper
        else:
            obj._model = wrapper

        return obj
