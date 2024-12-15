from typing import Type

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base_random_forest import BaseRandomForest
from .parameters import RFHyperparameters
from assignment2.util import get_rmse, convert_to_numpy


def run_random_forest(
        model_cls: Type[BaseRandomForest],
        parameters: RFHyperparameters,
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame,
        verbose: bool = False) -> dict[str, float]:
    """
    Fit and evaluate a random forest model with the given parameters

    Args:
        model_cls: Implementation of the random forest model
        parameters: Hyperparameters for the random forest model
        x_train: Training data
        x_test: Testing data
        y_train: Training labels
        y_test: Testing labels
        verbose: Whether to print a message at the start and end of the run

    Returns:
        Dictionary with the hyperparameters and the RMSE of the model
    """
    if verbose:
        print(f"Running with parameters: {parameters}")
    x_train = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else x_train
    x_test = x_test.to_numpy() if isinstance(x_test, pd.DataFrame) else x_test
    y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train
    y_test = y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test

    model = model_cls(
        data=np.column_stack((x_train, y_train)),
        no_of_trees=parameters.no_of_trees,
        max_depth=parameters.max_depth,
        min_samples=parameters.min_samples,
        feature_subset_size=parameters.feature_subset_size,
        task_type=parameters.task_type
    )
    model.fit()
    predictions = model.predict(x_test)
    rmse = model.evaluate(predictions, y_test)
    if verbose:
        print(f"Run complete with parameters {parameters}.\nRMSE: {rmse}")

    return {
        'trees': parameters.no_of_trees,
        'max_depth': parameters.max_depth,
        'min_samples': parameters.min_samples,
        'feature_subset_size': parameters.feature_subset_size,
        'RMSE': rmse
    }


def run_random_forest_with_varied_params(
        model_cls: Type[BaseRandomForest],
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame,
        hyperparameters: list[RFHyperparameters],
        verbose: bool = False,
        n_runs: int = 1,
        n_jobs: int = -1
) -> pd.DataFrame:
    """
    Fit and evaluate a random forest model with different hyperparameters

    Args:
        model_cls: Implementation of the random forest model
        x_train: Training data
        x_test: Testing data
        y_train: Training labels
        y_test: Testing labels
        hyperparameters: List of hyperparameters to test
        verbose: Whether to print a message at the start and end of each run
        n_runs: Number of times to run each set of hyperparameters
        n_jobs: Number of parallel jobs to run

    Returns:
        DataFrame with the RMSE for each set of hyperparameters
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_random_forest)(
            model_cls=model_cls,
            parameters=parameters,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            verbose=verbose
        ) for parameters in hyperparameters for _ in range(n_runs)
    )
    return pd.DataFrame(results)



def run_sklearn_model(
        model_cls: Type,
        parameters: dict[str, any],
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame,
        verbose: bool = False) -> dict[str, any]:
    if verbose:
        print(f"Running with parameters: {parameters}")

    x_train, x_test, y_train, y_test = convert_to_numpy(x_train, x_test, y_train, y_test)
    model = model_cls(**parameters)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = get_rmse(predictions, y_test)

    if verbose:
        print(f"Run complete with parameters {parameters}.\nRMSE: {rmse}")

    return parameters | { 'RMSE': rmse }


def run_sklearn_model_with_varied_params(
        model_cls: Type,
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame,
        hyperparameters: list[dict[str, any]],
        verbose: bool = False,
        n_runs: int = 1,
        n_jobs: int = -1
) -> pd.DataFrame:
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_sklearn_model)(
            model_cls=model_cls,
            parameters=parameters,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            verbose=verbose
        ) for parameters in hyperparameters for _ in range(n_runs)
    )
    return pd.DataFrame(results)
