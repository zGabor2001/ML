import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from assignment3.model.base_model import BaseRegressor


class LogisticRegressor(BaseRegressor):
    def __init__(self, input_dim, device=None):
        super(LogisticRegressor, self).__init__(device)
        self.model = nn.Linear(input_dim, 1).to(device)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              lr: float = 0.001):
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X_test: np.ndarray):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_test)
        return predictions.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test.to_numpy()) ** 2)
        print(f"Mean Squared Error: {mse:.4f}")
