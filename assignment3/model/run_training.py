import pandas as pd
import numpy as np
from assignment3.model.logistic_regressor import LogisticRegressor
from assignment3.model.lightgbm_regressor import LightGBMRegressor
from assignment3.model.xgboost_regressor import XGBoostRegressor
from assignment3.model.mlp_regressor import MLPRegressor
from assignment3.model.fnn_regressor import FNNRegressor
#from assignment3.automl.run_automl import run_automl_tables
from assignment3.toronto_rental import prepare_toronto_rental_dataset
from assignment3.energy_efficiency import prepare_energy_efficiency_dataset
from assignment3.employee_salaries import prepare_employee_salaries_dataset
from assignment3.cars import prepare_cars_dataset


def get_datasets():
    return {
        'toronto_rental': prepare_toronto_rental_dataset,
        'employee_salaries': prepare_employee_salaries_dataset,
        'energy_efficiency': prepare_energy_efficiency_dataset,
        'cars': prepare_cars_dataset
    }


def train_on_all_datasets():

    datasets: dict = get_datasets()

    results_on_all_datasets: list = []

    for name, dataset_func in datasets.items():
        print(f"\nTraining on dataset: {name}")

        x_train, x_test, y_train, y_test = dataset_func()

        #run_automl_tables('1', x_train, x_test, y_train, y_test)

        models = [
            {
                'dataset': name,
                'name': 'Logistic Regression',
                'model': LogisticRegressor(input_dim=x_train.shape[1]),
                'train_params': {'X_train': x_train, 'y_train': y_train, 'epochs': 100, 'batch_size': 32, 'lr': 0.001},
                'predict_params': {'X_test': x_test},
                'evaluate_params': {'y_test': y_test}
            },
            {
                'dataset': name,
                'name': 'XGBoost Regressor',
                'model': XGBoostRegressor(device='gpu'),
                'train_params': {'X_train': x_train, 'y_train': y_train, 'num_boost_round': 100},
                'predict_params': {'X_test': x_test},
                'evaluate_params': {'y_test': y_test}
            },
            {
                'dataset': name,
                'name': 'LightGBM Regressor',
                'model': LightGBMRegressor(device='gpu'),
                'train_params': {'X_train': x_train, 'y_train': y_train, 'num_boost_round': 100},
                'predict_params': {'X_test': x_test},
                'evaluate_params': {'y_test': y_test}
            },
            {
                'dataset': name,
                'name': 'FNN Regressor',
                'model': FNNRegressor(input_dim=x_train.shape[1]),
                'train_params': {'X_train': x_train, 'y_train': y_train, 'epochs': 100, 'batch_size': 32, 'lr': 0.001},
                'predict_params': {'X_test': x_test},
                'evaluate_params': {'y_test': y_test}
            },
            {
                'dataset': name,
                'name': 'MLP Regressor',
                'model': MLPRegressor(input_dim=x_train.shape[1]),
                'train_params': {'X_train': x_train, 'y_train': y_train, 'epochs': 100, 'batch_size': 32, 'lr': 0.001},
                'predict_params': {'X_test': x_test},
                'evaluate_params': {'y_test': y_test}
            }
        ]

        results_on_all_datasets.append(
            pd.DataFrame(run_training_on_preprocessed_dataset(models=models))
        )
    df_results: pd.DataFrame = pd.concat(results_on_all_datasets, axis=0)
    df_results.to_csv('train_results.csv')


def run_training_on_preprocessed_dataset(models: list) -> dict:
    results: dict = {'dataset': [], 'model': [], 'train_params': [], 'RMSE': []}
    for model_info in models:
        model = model_info['model']
        print(f"\nTraining {model_info['name']}...")
        model.train(**model_info['train_params'])
        predictions: np.ndarray = model.predict(**model_info['predict_params'])
        rmse = model.evaluate(predictions=predictions, **model_info['evaluate_params'])
        results['dataset'].append(model_info['dataset'])
        results['model'].append(model_info['name'])
        train_params: dict = {key: model_info['train_params'][key]
                              for key in model_info['train_params']
                              if key not in {'X_train', 'y_train'}}
        results['train_params'].append(train_params.copy())
        results['RMSE'].append(rmse)
    return results

