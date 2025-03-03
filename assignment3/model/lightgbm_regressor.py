import lightgbm as lgb
import numpy as np
from assignment3.model.base_model import BaseRegressor
from assignment3.util.data_utils import get_regression_performance_metrics


class LightGBMRegressor(BaseRegressor):
    def __init__(self, device='cpu'):
        super(LightGBMRegressor, self).__init__(device)
        self.model = None

    def train(self, X_train, y_train, num_boost_round=100):
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu' if self.device == 'cuda' else 'cpu'
        }
        self.model = lgb.train(params, train_data, num_boost_round=num_boost_round)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, predictions: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
        return get_regression_performance_metrics(y_true=y_test, y_pred=predictions)
