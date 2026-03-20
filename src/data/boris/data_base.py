from enum import Enum
import pandas as pd

class DatasetType(Enum):
    SuperCandels = 1
    BasicCandels = 2


class Dataset:

    def _Load_(self, ticker_name : str, dtype : DatasetType):
        try:
            path = PATH_TO_DATA + "/" + 
            data = pd.DataFrame("")
            self.inited = True
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
    
    def __init__(self, dtype :DatasetType, ticker_name : str, val_ratio=0.1, test_ratio=0.1):
        self.inited = False
        self.data_type = DatasetType
        self._Split_(self._Load_(ticker_name, dtype), val_ratio, test_ratio)

    def GetTrain(self) -> pd.DataFrame:
        return self.train
    
    def GetVal(self) -> pd.DataFrame:
        return self.val
    
    def GetTest(self) -> pd.DataFrame:
        return self.test
