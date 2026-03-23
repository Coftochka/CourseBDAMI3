from Base_model import BaseModel
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import json


class LightGBMModel(BaseModel):
    def __init__(self, input_size: int, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 31, max_depth: int = -1, min_child_samples: int = 20, subsample: float = 0.8, colsample_bytree: float = 0.8, reg_alpha: float = 0.0, reg_lambda: float = 1.0, horizon: int = 1, seq_len: int = 30):
        """
        input:
            input_size: number of features per timestep
            n_estimators: number of boosting rounds
            learning_rate: shrinkage rate
            num_leaves: max leaves per tree
            max_depth: max tree depth (-1 = unlimited)
            min_child_samples: min samples in a leaf
            subsample: row subsampling ratio per tree
            colsample_bytree: feature subsampling ratio per tree
            reg_alpha: L1 regularisation
            reg_lambda: L2 regularisation
            horizon: number of steps to predict (reserved for future use)
            seq_len: length of the input sequence window
        """
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
        self.horizon = horizon
        self.seq_len = seq_len

        self._model = None
        self.feature_names: list[str] = []
        self.history = {"train_loss": [], "val_loss": []}

    def _make_windows(self, X: np.ndarray, y: np.ndarray):
        """
        input:
            X, y: ndarray (n_timesteps, n_features) / (n_timesteps,)
        output:
            X_win: ndarray (n_windows, seq_len * n_features)  — flattened
            y_win: ndarray (n_windows,)
        """
        X_win, y_win = [], []
        for i in range(len(X) - self.seq_len + 1):
            X_win.append(X[i : i + self.seq_len].flatten())
            y_win.append(y[i + self.seq_len - 1])
        return np.array(X_win, dtype=np.float32), np.array(y_win, dtype=np.float32)

    def _build_feature_names(self, n_features: int) -> list[str]:
        names = []
        for t in range(self.seq_len):
            lag = self.seq_len - 1 - t
            for f in range(n_features):
                names.append(f"feat_{f}_t-{lag}")
        return names

    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, scheduler=None, epochs: int = None, batch_size: int = None, verbose: bool = True):
        """
        input:
            X, y: np.ndarray (n_timesteps, n_features) / (n_timesteps,)
            X_val, y_val: np.ndarray (n_timesteps, n_features) / (n_timesteps,)
            epochs: overrides n_estimators if provided
            verbose: print training log every 50 rounds
        """
        if epochs is not None:
            self.n_estimators = epochs

        X_w, y_w = self._make_windows(X, y)
        self.feature_names = self._build_feature_names(X.shape[1])

        has_val = X_val is not None and y_val is not None

        eval_sets  = [(X_w, y_w)]
        eval_names = ["training"]
        if has_val:
            X_val_w, y_val_w = self._make_windows(X_val, y_val)
            eval_sets.append((X_val_w, y_val_w))
            eval_names.append("valid_0")

        evals_record: dict = {}
        callbacks = [lgb.record_evaluation(evals_record)]
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))

        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
        )

        self._model.fit(
            X_w, y_w,
            eval_set=eval_sets,
            eval_names=eval_names,
            feature_name=self.feature_names,
            callbacks=callbacks,
        )

        self.history = {"train_loss": [], "val_loss": []}
        self.history["train_loss"] = evals_record.get("training", {}).get("binary_logloss", [])
        if has_val:
            self.history["val_loss"] = evals_record.get("valid_0", {}).get("binary_logloss", [])

    def plot_loss(self):
        train = self.history.get("train_loss", [])
        val   = self.history.get("val_loss", [])

        if not train:
            raise RuntimeError("fit model before plotting loss")

        iters = range(1, len(train) + 1)
        plt.figure(figsize=(9, 4))
        plt.plot(iters, train, label="Train loss")
        if val:
            plt.plot(iters, val, label="Val loss")
        plt.xlabel("Iteration")
        plt.ylabel("Binary logloss")
        plt.title("Loss by iteration")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_importance(self, max_features: int = 20):
        if self._model is None:
            raise RuntimeError("fit model before plotting importance")

        lgb.plot_importance(
            self._model,
            importance_type="gain",
            max_num_features=max_features,
            figsize=(9, 6),
            title="Feature importance (gain)",
        )
        plt.tight_layout()
        plt.show()

    def _to_df(self, X_w: np.ndarray):
        import pandas as pd
        return pd.DataFrame(X_w, columns=self.feature_names)

    def predict_proba(self, X) -> np.ndarray:
        """
        input:
            X : ndarray (n_timesteps, n_features)
        output:
            proba: ndarray (n_windows,) ∈ [0, 1]
        """
        if self._model is None:
            raise RuntimeError("model is not fitted yet, call fit() first")
        X_w, _ = self._make_windows(X, np.zeros(len(X)))
        return self._model.predict_proba(self._to_df(X_w))[:, 1].astype(np.float32)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """
        input:
            X: ndarray (n_timesteps, n_features)
            threshold : decision threshold
        output:
            pred: ndarray (n_windows,) int
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_last(self, X, threshold: float = 0.5) -> tuple[float, bool]:
        """
        input:
            X: ndarray (n_timesteps, n_features)
            threshold : decision threshold
        output:
            proba: float ∈ [0, 1]
            pred: bool
        """
        if self._model is None:
            raise RuntimeError("model is not fitted yet, call fit() first")
        if len(X) < self.seq_len:
            raise ValueError(f"need at least {self.seq_len} candles, got {len(X)}")

        window = X[-self.seq_len:].flatten().reshape(1, -1).astype(np.float32)
        proba = self._model.predict_proba(self._to_df(window))[0, 1]
        return float(proba), proba >= threshold

    def score(self, X, y) -> dict:
        """
        input:
            X : ndarray (n_timesteps, n_features)
            y : ndarray (n_timesteps,)
        output:
            dict with accuracy, f1, roc_auc
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        X_w, y_w = self._make_windows(X, y)
        proba = self._model.predict_proba(self._to_df(X_w))[:, 1].astype(np.float32)
        pred  = (proba >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y_w, pred),
            "f1":       f1_score(y_w, pred),
            "roc_auc":  roc_auc_score(y_w, proba),
        }

    def save(self, path: str):
        """
        input:
            path: str
        """
        if self._model is None:
            raise RuntimeError("model is not fitted yet, call fit() first")
        self._model.booster_.save_model(path + ".txt")
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
                "horizon":           self.horizon,
                "seq_len":           self.seq_len,
            },
            "feature_names": self.feature_names,
        }
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        """
        input:
            path: str
        """
        with open(path + ".json") as f:
            meta = json.load(f)

        model = cls(**meta["config"])
        model.feature_names = meta["feature_names"]

        booster = lgb.Booster(model_file=path + ".txt")
        model._model = lgb.LGBMClassifier()
        model._model._Booster = booster
        model._model.fitted_ = True
        model._model.classes_ = np.array([0, 1])
        model._model.n_classes_ = 2
        model._model.objective_ = "binary"
        return model