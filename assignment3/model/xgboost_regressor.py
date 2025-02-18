import xgboost as xgb
import numpy as np

from assignment3.model.base_model import BaseRegressor


class XGBoostRegressor(BaseRegressor):
    def __init__(self, device='cpu'):
        super(XGBoostRegressor, self).__init__(device)
        self.model = None

    def train(self, X_train, y_train, num_boost_round=100):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist' if self.device == 'cuda' else 'hist',
            'predictor': 'gpu_predictor' if self.device == 'cuda' else 'cpu_predictor'
        }
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        print(f"Mean Squared Error: {mse:.4f}")
