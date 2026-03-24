from dotenv import load_dotenv; 
load_dotenv()
import os
from moexalgo import session, Market, Ticker
import datetime as dt
from tickers import TICKERS
import pandas as pd
import sys

def GetFormattedDate(timestamp : dt.datetime):
    days = str(timestamp.day)
    if (len(days) < 2):
        days = '0' + days
    month = str(timestamp.month)
    if (len(month) < 2):
        month = '0' + month
    year = str(timestamp.year)
    return f"{year}-{month}-{days}"

def TickerFilename1(ticker, timestamp1:dt.datetime):
    return f"{ticker}_{GetFormattedDate(timestamp1)}.csv"

def TickerFilename2(ticker, timestamp1:dt.datetime, timestamp2:dt.datetime):
    return f"{ticker}_{GetFormattedDate(timestamp1)}_{GetFormattedDate(timestamp2)}.csv"

def TickerFilename2str(ticker, timestamp1:str, timestamp2:str):
    return f"{ticker}_{timestamp1}_{timestamp2}.csv"

def DataTransform(data_ob, data_tr, data_or):
    data_ob = data_ob.drop(columns=["systime"])
    data_ob.rename(columns=
                   {'val_b': 'val_b_obstat', 
                    'val_s': 'val_s_obstat', 
                    'vol_b': 'vol_b_obstat', 
                    'vol_s': 'vol_s_obstat'}, 
    inplace=True)
    
    data_tr = data_tr.drop(columns=["systime"])
    
    data_or = data_or.drop(columns=["systime"])
    data_or.rename(columns=
                   {'val_b': 'val_b_order', ''
                   'val_s': 'val_s_order', 
                   'vol_b': 'vol_b_order', 
                   'vol_s': 'vol_s_order'}, 
    inplace=True)

    data = pd.merge(data_ob, data_tr, on=("tradedate", "tradetime", "ticker"), how="inner")
    data = pd.merge(data,    data_ob, on=("tradedate", "tradetime", "ticker"), how="inner")

    return data

def LoadSuperCandels(ticker : Ticker, dateBegin : str, dateEnd : str) -> pd.DataFrame:
    data_ob = ticker.obstats(start=dateBegin, end=dateEnd)
    data_tr = ticker.tradestats(start=dateBegin, end=dateEnd)
    data_or = ticker.orderstats(start=dateBegin, end=dateEnd)
    if (max(len(data_ob), len(data_or), len(data_tr)) == 0):
        return None
    #print(data_or.columns, data_ob.columns, data_tr.columns)

    return DataTransform(data_ob, data_tr, data_or)

def LoadTicker(tickerSymb, dateBegin, dateEnd):
    batchLenDays = 30

    begin = dt.datetime.strptime(dateBegin, "%Y-%m-%d").date()
    end = dt.datetime.strptime(dateEnd, "%Y-%m-%d").date()

    ticker = Ticker(tickerSymb)

    while (begin <= end): 
        batch_start = GetFormattedDate(begin)
        batch_end = GetFormattedDate(begin + dt.timedelta(days=batchLenDays - 1))
        # -1 тк библиотка почему-то работает отрезками, а не полуинтвервалами
    
        begin += dt.timedelta(days=batchLenDays)

        data = LoadSuperCandels(ticker, batch_start, batch_end)
        
        name = f"./dataset/{tickerSymb}/frac_data/{TickerFilename2str(tickerSymb, batch_start, batch_end)}"
        if not (data is None):
            data.to_csv(name, index=False)
            print("loaded", batch_start, batch_end, name)
        else:
            print("empty", batch_start, batch_end)

MOEX_TOKEN = os.getenv("MOEX_TOKEN")

session.TOKEN = MOEX_TOKEN

#dataBegin = "2020-05-01"
#dataEnd   = "2026-04-01"
dataBegin = "2019-01-01"
dataEnd   = "2026-04-01"


if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("missing ticker argument"); 
    print ("VALUES:", sys.argv[0], sys.argv[1])
    LoadTicker(sys.argv[1], dataBegin, dataEnd)import os 
import pandas as pd 
import sys 

def MergeDatasets(ticker_symb):
    path = f"./dataset/{ticker_symb}/frac_data/"
    file_list = [path+f for f in os.listdir(path) if (f.endswith('.csv') and (f.startswith(ticker_symb)))]
    
    df_list = []
    for file in file_list:
        df = pd.read_csv(file, header=0)
        df['timestamp'] = pd.to_datetime(df['tradedate'] + ' ' + df['tradetime'])
     #   print(*df.columns)
        df = df.drop(columns=["tradetime", "tradedate"])
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')

    combined_df.sort_values('timestamp', ignore_index=True)
    print(combined_df)
    combined_df.to_csv(f"./dataset/{ticker_symb}/{ticker_symb}_SUPER_FULL.csv", index=False)

#MergeDatasets("YDEX")

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("missing ticker argument"); 
    MergeDatasets(sys.argv[1])