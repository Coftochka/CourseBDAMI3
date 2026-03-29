from Base_model import BaseModel, RegressionMetricsAccumulator
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import json
import copy
import pandas as pd
from typing import Optional, Dict, List


# X: (n_samples, seq_len, n_features) — как после prepare_windows / как у TorchBaseModel;
# внутри строки flatten в вектор фич для LGBM (+ ticker_id в pooled/finetune).


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
        input_size        : number of features per timestep (для SBER — schema.INPUT_SIZE)
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

    def _check_windowed_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n_samples, seq_len, n_features), got shape {X.shape}"
            )
        if X.shape[1] != self.seq_len:
            raise ValueError(
                f"X: expected seq_len={self.seq_len}, got {X.shape[1]}"
            )
        if X.shape[2] != self.input_size:
            raise ValueError(
                f"X: expected n_features={self.input_size}, got {X.shape[2]}"
            )
        return X

    def _rows_to_lgbm_matrix(
        self,
        X: np.ndarray,
        ticker_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Оконный батч → 2D матрица признаков для LGBM.
        Каждая строка: flatten окна, при pooled/finetune в конец — ticker_id.
        """
        X = self._check_windowed_X(X)
        n = X.shape[0]
        if self.mode != "single":
            if ticker_ids is None:
                raise ValueError(f"ticker_ids is required for mode='{self.mode}'")
            tid = np.asarray(ticker_ids, dtype=np.int64)
            if tid.shape != (n,):
                raise ValueError(
                    f"ticker_ids must be shape ({n},), got {tid.shape}"
                )
        else:
            tid = None

        rows = []
        for i in range(n):
            w = X[i].reshape(-1)
            if tid is not None:
                w = np.append(w, float(tid[i]))
            rows.append(w)
        return np.array(rows, dtype=np.float32)

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
        **kwargs,
    ):
        """
        X, y         : оконные — X (n_samples, seq_len, n_features), y (n_samples,).
        ticker_ids   : (n_samples,) int — для pooled / finetune.

        mode="single":
            fit(X, y, X_val=..., y_val=...)

        mode="pooled":
            fit(X, y, ticker_ids=ids, X_val=..., y_val=..., ticker_ids_val=ids_val)

        mode="finetune":
            Use pretrain() and finetune() instead of fit() directly.

        kwargs : игнорируются (совместимость с Torch fit: checkpoint_dir и т.д.).
        """
        if self.mode != "single":
            assert ticker_ids is not None, f"ticker_ids is required for mode='{self.mode}'"

        if epochs is not None:
            self.n_estimators = epochs

        X = self._check_windowed_X(X)
        y_w = np.asarray(y, dtype=np.float32)
        if y_w.shape != (X.shape[0],):
            raise ValueError(
                f"y must be shape ({X.shape[0]},), got {y_w.shape}"
            )

        with_ticker = ticker_ids is not None
        X_w = self._rows_to_lgbm_matrix(X, ticker_ids)
        self.feature_names = self._build_feature_names(self.input_size, with_ticker=with_ticker)

        X_val_w, y_val_w = None, None
        if X_val is not None and y_val is not None:
            X_val = self._check_windowed_X(X_val)
            y_val_w = np.asarray(y_val, dtype=np.float32)
            if y_val_w.shape != (X_val.shape[0],):
                raise ValueError(
                    f"y_val must be shape ({X_val.shape[0]},), got {y_val_w.shape}"
                )
            X_val_w = self._rows_to_lgbm_matrix(X_val, ticker_ids_val)

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

        X = self._check_windowed_X(X)
        n = X.shape[0]
        ids = np.full(n, ticker_id, dtype=np.int64)

        y_w = np.asarray(y, dtype=np.float32)
        if y_w.shape != (n,):
            raise ValueError(f"y must be shape ({n},), got {y_w.shape}")
        X_w = self._rows_to_lgbm_matrix(X, ids)

        X_val_w, y_val_w = None, None
        if X_val is not None and y_val is not None:
            X_val = self._check_windowed_X(X_val)
            ids_val = np.full(X_val.shape[0], ticker_id, dtype=np.int64)
            y_val_w = np.asarray(y_val, dtype=np.float32)
            if y_val_w.shape != (X_val.shape[0],):
                raise ValueError(
                    f"y_val shape mismatch: got {y_val_w.shape}, expected ({X_val.shape[0]},)"
                )
            X_val_w = self._rows_to_lgbm_matrix(X_val, ids_val)

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

    def predict(
        self,
        X,
        ticker_ids=None,
        use_finetune: bool = False,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """
        X              : (n_samples, seq_len, n_features) — как в fit
        ticker_ids     : (n_samples,) int — для pooled / finetune
        use_finetune   : if True, uses the fine-tuned model (mode="finetune" only)
        batch_size     : предсказание порциями (меньше пиковая память на больших n).

        Returns:
            pred : (n_samples,) float
        """
        model = self._active_model(finetune=use_finetune)
        X = self._check_windowed_X(X)
        n = X.shape[0]
        if self.mode != "single":
            if ticker_ids is None:
                raise ValueError(f"ticker_ids is required for mode='{self.mode}'")
            tid = np.asarray(ticker_ids, dtype=np.int64)
            if tid.shape != (n,):
                raise ValueError(
                    f"ticker_ids must be shape ({n},), got {tid.shape}"
                )
        else:
            tid = None
        outs: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X[start:end]
            ids_b = tid[start:end] if tid is not None else None
            X_w = self._rows_to_lgbm_matrix(Xb, ids_b)
            outs.append(model.predict(self._to_df(X_w)).astype(np.float32))
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs, axis=0)

    def predict_last(self, X, ticker_id: Optional[int] = None, use_finetune: bool = False) -> float:
        """
        X            : (seq_len, n_features), (1, seq_len, n_features) или (N, seq_len, n_features) — берётся X[-1]
        ticker_id    : int — для pooled / finetune
        use_finetune : if True, uses the fine-tuned model (mode="finetune" only)

        Returns:
            predicted return : float
        """
        model = self._active_model(finetune=use_finetune)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[-1]
        if X.ndim != 2:
            raise ValueError(f"predict_last: ожидается окно (seq_len, n_features), got {X.shape}")
        if X.shape != (self.seq_len, self.input_size):
            raise ValueError(
                f"predict_last: ожидается ({self.seq_len}, {self.input_size}), got {X.shape}"
            )

        window = X.reshape(-1)
        if self.mode != "single":
            if ticker_id is None:
                raise ValueError("ticker_id is required for pooled / finetune")
            window = np.append(window, float(ticker_id))
        window = window.reshape(1, -1).astype(np.float32)
        return float(model.predict(self._to_df(window))[0])

    @staticmethod
    def _regression_metrics_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> pd.DataFrame:
        acc = RegressionMetricsAccumulator()
        acc.update(y_true, y_pred)
        return acc.to_dataframe(model_name)

    def scores(
        self,
        X,
        y,
        model_name: str,
        ticker_ids=None,
        use_finetune: bool = False,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        """
        Regression metrics on the sample (батчи predict — меньше пиковая память).

        For multiple assets, pass ticker_ids — you'll get aggregated metrics.
        For metrics per asset separately, use scores_per_ticker().
        """
        X = self._check_windowed_X(X)
        n = X.shape[0]
        y_w = np.asarray(y, dtype=np.float32)
        if y_w.shape != (n,):
            raise ValueError(
                f"y must be shape ({n},), got {y_w.shape}"
            )
        if self.mode != "single":
            if ticker_ids is None:
                raise ValueError(f"ticker_ids is required for mode='{self.mode}'")
            tid = np.asarray(ticker_ids, dtype=np.int64)
            if tid.shape != (n,):
                raise ValueError(
                    f"ticker_ids must be shape ({n},), got {tid.shape}"
                )
        else:
            tid = None

        model = self._active_model(finetune=use_finetune)
        acc = RegressionMetricsAccumulator()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X[start:end]
            yb = y_w[start:end]
            ids_b = tid[start:end] if tid is not None else None
            X_w = self._rows_to_lgbm_matrix(Xb, ids_b)
            pred = model.predict(self._to_df(X_w)).astype(np.float32)
            acc.update(yb, pred)
        return acc.to_dataframe(model_name)

    def scores_per_ticker(
        self,
        X_dict: Dict[str, np.ndarray],
        y_dict: Dict[str, np.ndarray],
        ticker_to_id: Dict[str, int],
        use_finetune: bool = False,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        """
        X_dict       : тикер → X (n_samples, seq_len, n_features)
        y_dict       : тикер → y (n_samples,)
        ticker_to_id : {"AAPL": 0, "MSFT": 1, ...}

        Returns:
            DataFrame — one row per ticker + "ALL" (aggregate) row
        """
        frames: List[pd.DataFrame] = []
        all_acc = RegressionMetricsAccumulator()
        model = self._active_model(finetune=use_finetune)

        for ticker, X in X_dict.items():
            y = y_dict[ticker]
            X = self._check_windowed_X(X)
            ticker_ids = (
                None if self.mode == "single"
                else np.full(X.shape[0], ticker_to_id[ticker], dtype=np.int64)
            )
            y_w = np.asarray(y, dtype=np.float32)
            if y_w.shape != (X.shape[0],):
                raise ValueError(
                    f"{ticker}: y shape {y_w.shape} != ({X.shape[0]},)"
                )
            n = X.shape[0]
            if self.mode != "single":
                tid = np.asarray(ticker_ids, dtype=np.int64)
            else:
                tid = None
            acc = RegressionMetricsAccumulator()
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X[start:end]
                yb = y_w[start:end]
                ids_b = tid[start:end] if tid is not None else None
                X_w = self._rows_to_lgbm_matrix(Xb, ids_b)
                pred = model.predict(self._to_df(X_w)).astype(np.float32)
                acc.update(yb, pred)
            frames.append(acc.to_dataframe(ticker))
            all_acc.merge(acc)

        frames.append(all_acc.to_dataframe("ALL"))
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
