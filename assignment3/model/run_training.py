from assignment3.model.logistic_regressor import LogisticRegressor
from assignment3.model.lightgbm_regressor import LightGBMRegressor
from assignment3.model.xgboost_regressor import XGBoostRegressor
from assignment3.model.mlp_regressor import MLPRegressor
from assignment3.model.fnn_regressor import FNNRegressor


def run_training_on_preprocessed_dataset(x_train, x_test, y_train, y_test):
    logistic = LogisticRegressor()
    logistic.train(X_train=x_train,
                   y_train=y_train,
                   epochs=1,
                   batch_size=32,
                   lr=0.001)
    logistic.predict(x_test)
    logistic.evaluate(x_test, y_test)
