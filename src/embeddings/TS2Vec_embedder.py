import json
import os

from ts2vec import TS2Vec
from torch import cuda


class TS2VecEmbedder:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 320,
        device: str = "cpu",
        depth: int = 10,
    ):
        device = device or ("cuda" if cuda.is_available() else "cpu")
        self.config = dict(input_size=input_size, hidden_size=hidden_size, output_size=output_size, device=device, depth=depth)
        self.model = TS2Vec(
            input_dims=input_size, hidden_dims=hidden_size, output_dims=output_size, device=device, depth=depth
        )

    def fit_transform(self, X_train, n_epochs: int = 5, verbose: bool = True):
        self.model.fit(train_data=X_train, n_epochs=n_epochs, verbose=verbose)
        return self.model.encode(X_train, encoding_window='full_series')

    def transform(self, X):
        return self.model.encode(X, encoding_window='full_series')

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "weights.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)

    @classmethod
    def load(cls, path: str) -> "TS2VecEmbedder":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = cls(**config)
        obj.model.load(os.path.join(path, "weights.pt"))
        return obj
