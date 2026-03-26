"""
Скачивает super candles (obstats + tradestats + orderstats) параллельно.
Формат выходного CSV совпадает с SBER_SUPER_FULL.csv:
  ticker, spread_bbo_x, ..., vwap_s_1mio_x,
  pr_open, ..., pr_vwap_s,
  sec_pr_open, ..., sec_pr_close,
  spread_bbo_y, ..., vwap_s_1mio_y,
  timestamp
"""
import os
import sys
import time
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from moexalgo import session, Ticker

MOEX_TOKEN = Path(__file__).resolve().parents[2].joinpath('.env').read_text().split('MOEX_TOKEN=')[1].strip()

DATE_BEGIN  = "2020-07-06"   # данные obstats доступны с этой даты
DATE_END    = "2026-03-25"
BATCH_DAYS  = 90
WORKERS     = 12


def DataTransform(data_ob: pd.DataFrame, data_tr: pd.DataFrame, data_or: pd.DataFrame) -> pd.DataFrame:
    # Воспроизводим логику download_basic_candels.py строки 48-49 точно:
    # merge(ob, tr) → merge(result, ob) — отсюда суффиксы _x/_y в эталоне SBER
    data_ob = data_ob.drop(columns=["systime"])
    data_ob.rename(columns={
        'val_b': 'val_b_obstat',
        'val_s': 'val_s_obstat',
        'vol_b': 'vol_b_obstat',
        'vol_s': 'vol_s_obstat',
    }, inplace=True)

    data_tr = data_tr.drop(columns=["systime"])
    data_or = data_or.drop(columns=["systime"])

    data = pd.merge(data_ob, data_tr, on=("tradedate", "tradetime", "ticker"), how="inner")
    data = pd.merge(data,    data_ob, on=("tradedate", "tradetime", "ticker"), how="inner")

    data['timestamp'] = pd.to_datetime(data['tradedate'] + ' ' + data['tradetime'])
    data.drop(columns=["tradedate", "tradetime"], inplace=True)
    return data


def download_ticker(ticker_symb: str) -> tuple[str, str]:
    try:
        session.TOKEN = MOEX_TOKEN

        out_dir = Path(__file__).resolve().parent / "dataset" / ticker_symb
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{ticker_symb}_SUPER_FULL.csv"

        ticker = Ticker(ticker_symb)
        chunks = []

        begin = dt.date.fromisoformat(DATE_BEGIN)
        end   = dt.date.fromisoformat(DATE_END)

        while begin <= end:
            batch_end = min(begin + dt.timedelta(days=BATCH_DAYS - 1), end)
            bs = begin.isoformat()
            be = batch_end.isoformat()

            data_ob = ticker.obstats(start=bs,   end=be)
            data_tr = ticker.tradestats(start=bs, end=be)

            if len(data_ob) > 0 and len(data_tr) > 0:
                data_or = ticker.orderstats(start=bs, end=be)
                merged = DataTransform(data_ob, data_tr, data_or)
                chunks.append(merged)

            begin = batch_end + dt.timedelta(days=1)

        if chunks:
            combined = pd.concat(chunks, ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp'], keep='first')
            combined = combined.sort_values('timestamp', ignore_index=True)
            combined.to_csv(out_file, index=False)

            # HEAD50 для быстрого просмотра
            combined.head(50).to_csv(out_dir / f"{ticker_symb}_SUPER_HEAD50.csv", index=False)

            return ticker_symb, f"OK  — {len(combined)} rows → {out_file.name}"
        else:
            return ticker_symb, "EMPTY — no data"

    except Exception as e:
        return ticker_symb, f"ERROR — {e}"


def main(tickers_file: str):
    tickers = Path(tickers_file).read_text().strip().splitlines()
    tickers = [t.strip() for t in tickers if t.strip()]
    print(f"Скачиваем super candles для {len(tickers)} тикеров, {WORKERS} процессов\n")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_ticker, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            ticker, status = fut.result()
            print(f"[{time.time()-t0:6.1f}s] {ticker:12s} {status}")

    print(f"\nГотово за {time.time()-t0:.1f}s")


if __name__ == "__main__":
    tickers_file = sys.argv[1] if len(sys.argv) > 1 else "tickers_dem.txt"
    main(tickers_file)
