"""
Optuna hyperparameter search for all models, optimising IC on validation set.

Each run_optuna_<model>() returns (best_model, study, save_path).

Usage
-----
    from models.run_optuna import run_optuna_lstm
    model, study, path = run_optuna_lstm(X_train, y_train, X_val, y_val, n_trials=40)
    print(f"Best IC={study.best_value:.4f}  saved → {path}")
"""
from __future__ import annotations


import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch

from evaluation.metrics import regression_metrics
from models.LSTM_model import LSTMModel
from models.GRU_model import GRUModel
from models.CNN_model import CNNModel
from models.Transformer_model import TransformerModel
from models.LightGBM_model import LightGBMModel
from models.Arima_model import ArimaModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _val_ic(model, X_val, y_val) -> float:
    ic = float(regression_metrics(y_val, model.predict(X_val), model_name="tmp")["ic"].iloc[0])
    return ic if np.isfinite(ic) else -1.0


def _save(model, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        torch.save(model, path)
    else:
        path.write_bytes(pickle.dumps(model))
    print(f"Saved → {path}")
    return str(path)


# ── LSTM ───────────────────────────────────────────────────────────────────────

def run_optuna_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    save_dir: str = "experiments/models",
) -> Tuple[LSTMModel, optuna.Study, str]:
    input_size = X_train.shape[2]

    def objective(trial: optuna.Trial) -> float:
        m = LSTMModel(
            input_size=input_size,
            hidden_size=trial.suggest_int("hidden_size", 32, 256, step=32),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_int("epochs", 20, 80, step=10),
            batch_size=128,
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            patience=trial.suggest_int("patience", 5, 15),
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return _val_ic(m, X_val, y_val)

    study = optuna.create_study(direction="maximize", study_name="lstm_ic")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    p = study.best_trial.params
    best = LSTMModel(input_size=input_size, hidden_size=p["hidden_size"],
                     num_layers=p["num_layers"], dropout=p["dropout"],
                     epochs=p["epochs"], batch_size=128,
                     lr=p["lr"], patience=p["patience"])
    best.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return best, study, _save(best, Path(save_dir) / "lstm_best.pth")


# ── GRU ───────────────────────────────────────────────────────────────────────

def run_optuna_gru(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    save_dir: str = "experiments/models",
) -> Tuple[GRUModel, optuna.Study, str]:
    input_size = X_train.shape[2]

    def objective(trial: optuna.Trial) -> float:
        m = GRUModel(
            input_size=input_size,
            hidden_size=trial.suggest_int("hidden_size", 32, 256, step=32),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_int("epochs", 20, 80, step=10),
            batch_size=128,
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            patience=trial.suggest_int("patience", 5, 15),
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return _val_ic(m, X_val, y_val)

    study = optuna.create_study(direction="maximize", study_name="gru_ic")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    p = study.best_trial.params
    best = GRUModel(input_size=input_size, hidden_size=p["hidden_size"],
                    num_layers=p["num_layers"], dropout=p["dropout"],
                    epochs=p["epochs"], batch_size=128,
                    lr=p["lr"], patience=p["patience"])
    best.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return best, study, _save(best, Path(save_dir) / "gru_best.pth")


# ── CNN ───────────────────────────────────────────────────────────────────────

def run_optuna_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    save_dir: str = "experiments/models",
) -> Tuple[CNNModel, optuna.Study, str]:
    input_size = X_train.shape[2]

    def objective(trial: optuna.Trial) -> float:
        m = CNNModel(
            input_size=input_size,
            num_filters=trial.suggest_int("num_filters", 32, 256, step=32),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            kernel_size=trial.suggest_int("kernel_size", 3, 9, step=2),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_int("epochs", 20, 80, step=10),
            batch_size=128,
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            patience=trial.suggest_int("patience", 5, 15),
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return _val_ic(m, X_val, y_val)

    study = optuna.create_study(direction="maximize", study_name="cnn_ic")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    p = study.best_trial.params
    best = CNNModel(input_size=input_size, num_filters=p["num_filters"],
                    num_layers=p["num_layers"], kernel_size=p["kernel_size"],
                    dropout=p["dropout"], epochs=p["epochs"],
                    batch_size=128, lr=p["lr"], patience=p["patience"])
    best.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return best, study, _save(best, Path(save_dir) / "cnn_best.pth")


# ── Transformer ───────────────────────────────────────────────────────────────

def run_optuna_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    save_dir: str = "experiments/models",
) -> Tuple[TransformerModel, optuna.Study, str]:
    input_size = X_train.shape[2]

    def objective(trial: optuna.Trial) -> float:
        d_model = trial.suggest_int("d_model", 32, 256, step=32)
        nhead = trial.suggest_categorical("nhead", [h for h in [2, 4, 8] if d_model % h == 0])
        m = TransformerModel(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=trial.suggest_int("num_encoder_layers", 1, 4),
            dim_feedforward=trial.suggest_int("dim_feedforward", 64, 512, step=64),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_int("epochs", 20, 80, step=10),
            batch_size=128,
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            patience=trial.suggest_int("patience", 5, 15),
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return _val_ic(m, X_val, y_val)

    study = optuna.create_study(direction="maximize", study_name="transformer_ic")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    p = study.best_trial.params
    best = TransformerModel(input_size=input_size, d_model=p["d_model"],
                            nhead=p["nhead"], num_encoder_layers=p["num_encoder_layers"],
                            dim_feedforward=p["dim_feedforward"], dropout=p["dropout"],
                            epochs=p["epochs"], batch_size=128,
                            lr=p["lr"], patience=p["patience"])
    best.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return best, study, _save(best, Path(save_dir) / "transformer_best.pth")


# ── LightGBM ─────────────────────────────────────────────────────────────────

def run_optuna_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    save_dir: str = "experiments/models",
) -> Tuple[LightGBMModel, optuna.Study, str]:

    def objective(trial: optuna.Trial) -> float:
        m = LightGBMModel(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, step=16),
            max_depth=trial.suggest_int("max_depth", -1, 14),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80, step=5),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
            random_state=42,
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return _val_ic(m, X_val, y_val)

    study = optuna.create_study(direction="maximize", study_name="lightgbm_ic")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    p = study.best_trial.params
    best = LightGBMModel(**p, random_state=42)
    best.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return best, study, _save(best, Path(save_dir) / "lightgbm_best.pkl")


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def run_optuna_arima(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 30,
    save_dir: str = "experiments/models",
) -> Tuple[ArimaModel, optuna.Study, str]:

    def objective(trial: optuna.Trial) -> float:
        m = ArimaModel(
            p=trial.suggest_int("p", 0, 5),
            d=trial.suggest_int("d", 0, 2),
            q=trial.suggest_int("q", 0, 5),
            trend=trial.suggest_categorical("trend", ["n", "c"]),
            auto_order=False,
        )
        m.fit(X_train, y_train)
        return _val_ic(m, X_val, y_val)

    study = optuna.create_study(direction="maximize", study_name="arima_ic")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    p = study.best_trial.params
    best = ArimaModel(p=p["p"], d=p["d"], q=p["q"], trend=p["trend"], auto_order=False)
    best.fit(X_train, y_train)
    return best, study, _save(best, Path(save_dir) / "arima_best.pkl")
