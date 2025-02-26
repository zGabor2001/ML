import os
from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
from datetime import timedelta

from assignment3.cars import prepare_cars_dataset
from assignment3.employee_salaries import prepare_employee_salaries_dataset
from assignment3.energy_efficiency import prepare_energy_efficiency_dataset
from assignment3.model.fnn_regressor import FNNRegressor
from assignment3.model.lightgbm_regressor import LightGBMRegressor
from assignment3.model.logistic_regressor import LogisticRegressor
from assignment3.model.mlp_regressor import MLPRegressor
from assignment3.model.xgboost_regressor import XGBoostRegressor
from assignment3.simulated_annealing import ModelConfig, SimulatedAnnealing
from assignment3.toronto_rental import prepare_toronto_rental_dataset


@dataclass
class Dataset:
    name: str
    folder: str
    prepare_func: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def main():
    datasets = [
        Dataset(
            name='Cars',
            folder='cars',
            prepare_func=prepare_cars_dataset,
        ),
        Dataset(
            name='Employee Salaries',
            folder='employee_salaries',
            prepare_func=prepare_employee_salaries_dataset,
        ),
        Dataset(
            name='Energy Efficiency',
            folder='energy_efficiency',
            prepare_func=prepare_energy_efficiency_dataset
        ),
        Dataset(
            name='Toronto Rental',
            folder='toronto_rental',
            prepare_func=prepare_toronto_rental_dataset
        ),
    ]

    models = [
        ModelConfig(
            name="FNN Regressor",
            model_cls=FNNRegressor,
            parameters={
                "epochs": list(range(50, 500, 10)),
                "batch_size": [16, 32, 64, 128, 256],
                "lr": [
                    1e-4, 3e-4,
                    1e-3, 3e-3,
                    1e-2, 3e-2
                ]
            }
        ),
        ModelConfig(
            name="LightGBM Regressor",
            model_cls=LightGBMRegressor,
            parameters={
                "num_boost_round": list(range(50, 2000, 1))
            }
        ),
        ModelConfig(
            name="Logistic Regressor",
            model_cls=LogisticRegressor,
            parameters={
                'epochs': list(range(50, 500, 10)),
                'batch_size': [16, 32, 64, 128, 256],
                'lr': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
            }
        ),
        ModelConfig(
            name="MLP Regressor",
            model_cls=MLPRegressor,
            parameters={
                'epochs': list(range(50, 500, 10)),
                'batch_size': [32, 64, 128, 256],
                'lr': [1e-4, 3e-4, 1e-3, 3e-3],
            }
        ),
        ModelConfig(
            name="XGBoost Regressor",
            model_cls=XGBoostRegressor,
            parameters={
                'num_boost_round': list(range(50, 2000, 1))
            }
        )
    ]

    for dataset in datasets:
        print(f"Preparing dataset {dataset.name}")
        x_train, x_test, y_train, y_test = dataset.prepare_func()
        print(f"Running simulated annealing on dataset {dataset.name} with hyperparameters {models}")
        simulated_annealing = SimulatedAnnealing(
            model_configs=models,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            initial_acceptance_rate = 0.99,
            p_test_different_model = 0.2,
            neighbor_range = 0.5,
            max_time = timedelta(hours=2)
        )
        simulated_annealing.run()
        best_solution = simulated_annealing.best_solution
        print(f"Best solution for dataset {dataset.name}: {best_solution.to_dict()}")
        solutions_history = simulated_annealing.solutions_history_df

        os.makedirs(f"results/{dataset.folder}", exist_ok=True)
        solutions_history.to_csv(f"results/{dataset.folder}/simulated_annealing_results.csv")

if __name__ == "__main__":
    main()
