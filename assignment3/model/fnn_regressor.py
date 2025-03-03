import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from assignment3.model.base_model import BaseRegressor
from assignment3.util.data_utils import get_regression_performance_metrics


class FNNRegressor(BaseRegressor):
    def __init__(self, device=None):
        super(FNNRegressor, self).__init__(device)
        self.model = None
        self.device = device

    def train(self, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
        input_dim = X_train.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

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

    def predict(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_test)
        return predictions.cpu().numpy()

    def evaluate(self, predictions: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
        return get_regression_performance_metrics(y_true=y_test, y_pred=predictions)
