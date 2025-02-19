from autogluon.tabular import TabularDataset, TabularPredictor
from flaml import AutoML


def run_automl_tables(label, train_data, test_data):
    predictor = TabularPredictor(label=label).fit(
        train_data,
        num_gpus=1
    )

    autogluon_predictions = predictor.predict(test_data)

    automl = AutoML()
    automl_settings = {
        "time_budget": 60,
        "task": 'classification',
        "log_file_name": "flaml.log",
        "use_gpu": True
    }

    automl.fit(X_train=train_data.drop(columns=['target']), y_train=train_data['target'], **automl_settings)

    flaml_predictions = automl.predict(test_data.drop(columns=['target']))

    return autogluon_predictions, flaml_predictions
