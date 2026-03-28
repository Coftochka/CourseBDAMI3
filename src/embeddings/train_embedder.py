"""
Train LSTMEmbedder and save embeddings + model weights.

Usage (from project root):
    python -m src.embeddings.train_embedder --config experiments/configs/exp_001_k3_lstm.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# ── path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "data" / "data_analisys&prep"))

from LSTM_embedder import LSTMEmbedder  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_windows(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed window arrays.

    Expected keys in cfg["data"]:
        X_train, y_train, X_val, y_val  — paths to .npy files
    """
    data = cfg["data"]
    X_train = np.load(data["X_train"])
    y_train = np.load(data["y_train"])
    X_val   = np.load(data["X_val"])
    y_val   = np.load(data["y_val"])
    return X_train, y_train, X_val, y_val


def train(cfg: dict) -> LSTMEmbedder:
    X_train, y_train, X_val, y_val = _load_windows(cfg)

    emb_cfg = cfg["embedder"]
    model = LSTMEmbedder(
        input_size  = X_train.shape[-1],
        hidden_size = emb_cfg.get("hidden_size", 64),
        num_layers  = emb_cfg.get("num_layers", 2),
        dropout     = emb_cfg.get("dropout", 0.1),
    )

    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs     = emb_cfg.get("epochs", 50),
        batch_size = emb_cfg.get("batch_size", 512),
    )

    out = cfg["output"]
    Path(out["model"]).parent.mkdir(parents=True, exist_ok=True)
    model.save(out["model"])
    print(f"Model saved → {out['model']}")

    for split, X, label in [("train", X_train, "train_labels"), ("val", X_val, "val_labels")]:
        emb = model.transform(X)
        emb_path = out[f"emb_{split}"]
        Path(emb_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb)
        print(f"Embeddings ({split}) saved → {emb_path}  shape={emb.shape}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTMEmbedder")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
