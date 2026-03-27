from dotenv import load_dotenv; 
from pathlib import Path

PATH_TO_DATA="." # чтобы подавить подчеркивание. подтягивается из .env
env_path = Path(__file__).resolve().parents[3] / '.env'
load_dotenv(dotenv_path=env_path)

from enum import Enum
import pandas as pd

class DatasetType(Enum):
    SuperCandels = 1
    BasicCandels = 2
    
    def getFilename(self, ticker_name: str) -> str:
        if self == DatasetType.SuperCandels:
            return f"{ticker_name}_SUPER_FULL.csv"
        elif self == DatasetType.BasicCandels:
            return f"{ticker_name}_BASIC_FULL.parquet"
        return f"{ticker_name}.csv"

    def getLoader(self):
        if self == DatasetType.SuperCandels:
            return pd.read_csv
        elif self == DatasetType.BasicCandels:
            return pd.read_parquet
        return
    

class Dataset:
    def _Load_(self, ticker_name : str, dtype : DatasetType):
        try:

            filename = dtype.getFilename(ticker_name)
            file_path = Path(PATH_TO_DATA) / ticker_name / filename
            
            if dtype == DatasetType.SuperCandels:
                df = pd.read_csv(file_path, header=0, parse_dates=["timestamp"])
            elif dtype == DatasetType.BasicCandels:
                df = pd.read_parquet(file_path)        
            return df
        
        except Exception as e:
            print(e)
            return None
        
    def _Split_(self, data, val_ratio:float, test_ratio:float):
        train_ratio = 1 - val_ratio - test_ratio

        mx = max(train_ratio, val_ratio, test_ratio)
        mn = min(train_ratio, val_ratio, test_ratio)

        if (mn < 0 or mx > 1):
            raise Exception(f"Wrong Data Ratio : train{train_ratio} val {val_ratio} test {test_ratio}")

        n = len(data)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        self.train = data[:train_end]
        self.val = data[train_end:val_end]
        self.test = data[val_end:]
        return 
    
    def __init__(self, dtype :DatasetType, ticker_name : str, val_ratio=0.2, test_ratio=0.2):
        self.inited = False
        self.data_type = DatasetType
        self._Split_(self._Load_(ticker_name, dtype), val_ratio, test_ratio)
    
    def ApplyTransform(self, func):
        self.train = func(self.train)
        self.val = func(self.val)
        self.test = func(self.val)  
        return None

    def GetTrain(self) -> pd.DataFrame:
        return self.train
    
    def GetVal(self) -> pd.DataFrame:
        return self.val
    
    def GetTest(self) -> pd.DataFrame:
        return self.test

