from logistic_regressor import LogisticRegressor
from lightgbm_regressor import LightGBMRegressor
from xgboost_regressor import XGBoostRegressor
from mlp_regressor import MLPRegressor
from fnn_regressor import FNNRegressor


def run_training_on_preprocessed_dataset(x_train, x_test, y_train, y_test):
    logistic = LogisticRegressor()
    logistic.train(X_train=x_train,
                   y_train=y_train,
                   epochs=1,
                   batch_size=32,
                   lr=0.001
                  )
    logistic.predict(x_test)
    logistic.evaluate(x_test, y_test)
