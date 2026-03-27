from dotenv import load_dotenv; 
load_dotenv()
import os
from moexalgo import session, Market, Ticker
import datetime as dt
from tickers import TICKERS
import pandas as pd
import sys

def GetForattedDate(timestamp):
    pass

def TickerFilename2str(ticker, timestamp1:str, timestamp2:str):
    return f"{ticker}_{timestamp1}_{timestamp2}.csv"

def DataTransform(data_ob : pd.DataFrame, data_tr : pd.DataFrame, data_or : pd.DataFrame):
    print(len(data_ob), len(data_tr), len(data_or))

    data_ob['timestamp'] = pd.to_datetime(data_ob['tradedate'] + ' ' + data_ob['tradetime'])
    data_ob = data_ob.drop(columns=["tradedate", "tradetime", "systime"])
    data_ob.set_index("timestamp")

    data_or['timestamp'] = pd.to_datetime(data_or['tradedate'] + ' ' + data_or['tradetime'])
    data_or = data_or.drop(columns=["tradedate", "tradetime", "systime"])
    data_or.set_index("timestamp")

    data_tr['timestamp'] = pd.to_datetime(data_tr['tradedate'] + ' ' + data_tr['tradetime'])
    data_tr = data_tr.drop(columns=["tradedate", "tradetime", "systime"])
    data_tr.set_index("timestamp")

    data_ob.rename(columns=
                   {'val_b': 'val_b_obstat', 
                    'val_s': 'val_s_obstat', 
                    'vol_b': 'vol_b_obstat', 
                    'vol_s': 'vol_s_obstat'}, 
    inplace=True)
    

    data_or.rename(columns=
                   {'val_b': 'val_b_order', ''
                   'val_s': 'val_s_order', 
                   'vol_b': 'vol_b_order', 
                   'vol_s': 'vol_s_order'}, 
    inplace=True)

    data = pd.concat([
        data_ob[["timestamp", "ticker"]],
        data_or[["timestamp", "ticker"]],
        data_tr[["timestamp", "ticker"]]
    ])
    
    # Удаление дубликатов и сортировка
    data = data.drop_duplicates(subset=["timestamp", "ticker"])
    data = data.sort_values("timestamp", ascending=True).reset_index(drop=True)
    data.set_index("timestamp")
    
    data = pd.merge(data, data_ob, on=("timestamp", "ticker"), how="left")
    data = pd.merge(data, data_or, on=("timestamp", "ticker"), how="left")
    data = pd.merge(data, data_tr, on=("timestamp", "ticker"), how="left")
     
    print(len(data))
    return data

def LoadSuperCandels(ticker : Ticker, dateBegin : str, dateEnd : str) -> pd.DataFrame:
    
    data_ob = ticker.obstats(start=dateBegin, end=dateEnd, )
    data_tr = ticker.tradestats(start=dateBegin, end=dateEnd)
    data_or = ticker.orderstats(start=dateBegin, end=dateEnd)
   
    return DataTransform(data_ob, data_tr, data_or)


def LoadTicker(tickerSymb, dateBegin, dateEnd) -> list[pd.DataFrame]:

    ticker = Ticker(tickerSymb)
    begin = ""
    files = sorted(os.listdir(f"./dataset/{tickerSymb}/frac_data/"))
    
    for file in files: 
        if (tickerSymb in file) and not ("SUPER_FULL" in file) and ".csv" in file:
            begin = file

    #нашли файл частичного сохранения 
    if (begin != ""):
        begin = begin.split("_")[-1].split('.')[0]

    #только начинаем загрузку
    else:
        begin = dateBegin

    begin0 = ""
    while (begin <= dateEnd) and (begin0 != begin): 
        begin0 = begin

        data = LoadSuperCandels(ticker, begin, "today")        
        begin = data["timestamp"].max().strftime('%Y-%m-%d') 

        if not (data is None):
            name = f"./dataset/{tickerSymb}/frac_data/{TickerFilename2str(tickerSymb, begin0, begin)}"
            print("loaded", begin0, begin, name)
            data.to_csv(name, index=False)
            
        else:
            print("empty", begin0, begin)
    return



MOEX_TOKEN = os.getenv("MOEX_TOKEN")

session.TOKEN = MOEX_TOKEN

dataBegin = "2001-01-01"
dataEnd   = "2026-03-24" # ДОЛЖНО БЫТ ЬСТРОГО МЕНЬШЕ СЕГОДНЯШНЕГО ДНЯ 

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("missing ticker argument"); 
    print ("VALUES:", sys.argv[0], sys.argv[1])
    LoadTicker(sys.argv[1], dataBegin, dataEnd)