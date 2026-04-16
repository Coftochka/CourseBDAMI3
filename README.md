# Financial Time Series Analysis and ML

Prediction and optimization algorithms for financial time series.  
The main code is located in `src`.

## Structure

- `src/data` - data loading and preprocessing (candles, tickers, dataloader).
- `src/models` - model implementations (`CNN`, `GRU`, `LSTM`, `Transformer`, `LightGBM`, `ARIMA`, etc.), plus `run_optuna.py` for hyperparameter tuning.
- `src/embeddings` - embedders: `Handmade_embedder.py` and `TS2Vec_embedder.py`.
- `src/evaluation` - metrics and model comparison.
- `src/clustering` - `KMeans`, `HDBSCAN`, `UMAP`, and cluster analysis.
- `src/loaded_models` - saved weights:
  - `full_data_models` - main best-model checkpoints;
  - `ts2vec` - TS2Vec weights and configs (`daily`/`hourly`);
  - `ts2vec_models` - model weights for each cluster produced by TS2Vec + KMeans;
  - `handmade_models` - model weights for each cluster produced by handmade features + KMeans.
- `moex_filled` - prepared data (if already downloaded/filled).
- `experiments` - experimental work and intermediate results.

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r reqirements.txt
   ```

2. Prepare environment:
   - copy `.env_example` to `.env` if needed;
   - check data/model paths in scripts you use.

3. Use modules from `src` as a Python package and load pretrained weights from `src/loaded_models`.

4. For experiments, run hyperparameter tuning via `src/models/run_optuna.py` and compare results with modules from `src/evaluation`.

## Notes

- The project includes both full checkpoints (`full_data_models`) and lighter cluster-specific files in `ts2vec_models` and `handmade_models`.
- If weights fail to load, verify filename and model architecture compatibility.