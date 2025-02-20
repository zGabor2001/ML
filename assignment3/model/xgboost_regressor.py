import xgboost as xgb

from assignment3.model.base_model import BaseRegressor
from assignment3.util.data_utils import get_rmse


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

    def predict(self, X_test: np.ndarray):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)

    def evaluate(self, predictions: np.ndarray, y_test):
        rmse = get_rmse()
        print(f"Mean Squared Error: {rmse:.4f}")
