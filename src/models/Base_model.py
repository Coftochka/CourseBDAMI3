from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    
    @abstractmethod
    def fit(self, X, y, **kwargs): ...
    
    @abstractmethod
    def predict(self, X): ...
    
    @abstractmethod
    def save(self, path: str): ...
    
    @abstractmethod
    def load(self, path: str): ...