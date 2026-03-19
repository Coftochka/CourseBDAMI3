from dotenv import load_dotenv; 
load_dotenv()
import os
from moexalgo import session, Market, Ticker
import datetime as dt
from tickers import TICKERS


def getFormattedDate(timestamp : dt.datetime):
    days = str(timestamp.day)
    if (len(days) < 2):
        days = '0' + days
    month = str(timestamp.month)
    if (len(month) < 2):
        month = '0' + month
    year = str(timestamp.year)
    return f"{year}-{month}-{days}"

def TickerName1(ticker, timestamp1:dt.datetime):
    return f"{ticker}_{getFormattedDate(timestamp1)}.csv"

def TickerName2(ticker, timestamp1:dt.datetime, timestamp2:dt.datetime):
    return f"{ticker}_{getFormattedDate(timestamp1)}.csv"



MOEX_TOKEN = os.getenv("MOEX_TOKEN")

session.TOKEN = MOEX_TOKEN

dataBegin = "2023-01-01"
dataEnd = "2023-01-15"

def get_day(ticker, dateBegin, dateEnd, filename):
    data = ticker.obstats(start=dateBegin, end=dateEnd)
    data = ticker.candles(
                start=dateBegin, end=dateEnd, period=1)
    data.to_csv(filename)

def load_ticker(tickerSymb, dateBegin, dateEnd):
    begin = dt.datetime.strptime(dateBegin, "%Y-%m-%d").date()
    end = dt.datetime.strptime(dateEnd, "%Y-%m-%d").date()
    ticker = Ticker(tickerSymb)

    while (begin <= end): 
        name = f"./{tickerSymb}/{TickerName1(tickerSymb, begin)}"
        cur = getFormattedDate(begin)
        print(f"loading {cur}",end=" ")
        begin += dt.timedelta(days=1)
        curp1 = getFormattedDate(begin)

        get_day(ticker, cur, curp1, name)
        print("loaded;")
    
load_ticker("SBER", dataBegin, dataEnd)