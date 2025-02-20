from autogluon.tabular import TabularDataset, TabularPredictor
from flaml import AutoML
import pandas as pd


def run_automl_tables(target_variable, X_train, y_train, X_test, y_test):
    train_data = pd.concat([X_train, y_train], axis=1)

    predictor = TabularPredictor(label=target_variable).fit(
        train_data,
        num_gpus=1
    )

    autogluon_predictions = predictor.predict(X_test)

    automl = AutoML()
    automl_settings = {
        "time_budget": 60,
        "task": 'classification',
        "log_file_name": "flaml.log",
        "use_gpu": True
    }

    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    flaml_predictions = automl.predict(X_test)

    return autogluon_predictions, flaml_predictions
