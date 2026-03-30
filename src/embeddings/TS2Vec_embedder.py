from ts2vec import TS2Vec
from torch import cuda
import torch


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
        return self.model.encode(X_train)

    def transform(self, X):
        return self.model.encode(X)

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "config": dict(self.config),
        }, path)

    @classmethod
    def load(cls, path: str) -> "TS2VecEmbeder":
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        obj = cls(**ckpt["config"])
        obj.model.load_state_dict(ckpt["state_dict"])
        return obj
