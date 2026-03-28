"""
src.embeddings
--------------
Re-exports LSTMEmbedder from its canonical source location so the rest
of the project can do:

    from src.embeddings import LSTMEmbedder
"""
import sys
from pathlib import Path

_prep_dir = Path(__file__).parent.parent / "data" / "data_analisys&prep"
if str(_prep_dir) not in sys.path:
    sys.path.insert(0, str(_prep_dir))

from LSTM_embedder import LSTMEmbedder  # noqa: E402

__all__ = ["LSTMEmbedder"]
