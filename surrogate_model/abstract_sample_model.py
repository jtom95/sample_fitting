from abc import ABC, abstractmethod

import numpy as np

class AbstractSampleModel(ABC):   
    @abstractmethod
    def predict(self, X):
        pass
    
    def prediction_std(self, X):
        return np.zeros_like(self.predict(X))