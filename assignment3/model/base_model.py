from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    def __init__(self, device):
        self.device = device  # To handle GPU / CPU support
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass
