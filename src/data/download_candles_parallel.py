"""
Скачивает 5-минутные OHLCV свечи (candles) для списка тикеров параллельно.
Сохраняет в dataset/<TICKER>/<TICKER>_BASIC_FULL.parquet
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def _load_token() -> str:
    env_file = Path(__file__).resolve().parents[2] / '.env'
    for line in env_file.read_text().splitlines():
        if line.startswith("MOEX_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("MOEX_TOKEN not found in .env")

import pandas as pd
from moexalgo import session, Ticker

MOEX_TOKEN = _load_token()

DATE_BEGIN = "2012-01-01"
DATE_END   = "2026-03-25"
PERIOD     = "5min"
WORKERS    = 6  # параллельных процессов


def download_ticker(ticker_symb: str) -> tuple[str, str]:
    """Скачивает все свечи для одного тикера, возвращает (ticker, статус)."""
    try:
        session.TOKEN = MOEX_TOKEN

        out_dir = Path(__file__).resolve().parent / "dataset" / ticker_symb
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{ticker_symb}_BASIC_FULL.parquet"

        t = Ticker(ticker_symb)
        chunks = []
        batch_days = 180  # дней за один запрос

        import datetime as dt
        begin = dt.date.fromisoformat(DATE_BEGIN)
        end   = dt.date.fromisoformat(DATE_END)

        while begin <= end:
            batch_end = min(begin + dt.timedelta(days=batch_days - 1), end)
            try:
                df = t.candles(
                    start=begin.isoformat(),
                    end=batch_end.isoformat(),
                    period=PERIOD,
                )
                if df is not None and len(df) > 0:
                    chunks.append(df)
            except Exception as e:
                pass  # пустой период — пропускаем
            begin = batch_end + dt.timedelta(days=1)

        if chunks:
            combined = pd.concat(chunks, ignore_index=True)
            combined = combined.drop_duplicates(subset=["begin"])
            combined = combined.sort_values("begin").reset_index(drop=True)
            combined.to_parquet(out_file, index=False)
            return ticker_symb, f"OK  — {len(combined)} rows → {out_file.name}"
        else:
            return ticker_symb, "EMPTY — no data"

    except Exception as e:
        return ticker_symb, f"ERROR — {e}"


def main(tickers_file: str):
    tickers = Path(tickers_file).read_text().strip().splitlines()
    tickers = [t.strip() for t in tickers if t.strip()]
    print(f"Скачиваем {len(tickers)} тикеров, {WORKERS} параллельных процессов\n")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_ticker, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            ticker, status = fut.result()
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s] {ticker:12s} {status}")

    print(f"\nГотово за {time.time() - t0:.1f}s")


if __name__ == "__main__":
    tickers_file = sys.argv[1] if len(sys.argv) > 1 else "tickers_dem.txt"
    main(tickers_file)
