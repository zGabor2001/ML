import numpy as np
import openml
import pandas as pd
import time
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score
from functools import wraps
from sklearn.model_selection import train_test_split


def get_dataset_from_openml(dataset_id: int) -> pd.DataFrame:
    """
       Fetches a dataset from OpenML and returns it as a pandas DataFrame.

       Parameters:
           dataset_id (int): The unique ID of the dataset to be fetched from OpenML.

       Returns:
           pd.DataFrame: A pandas DataFrame containing the data from the requested OpenML dataset.
       """
    dataset = openml.datasets.get_dataset(dataset_id)
    # we will perform any data splits manually, so we are only interested in the first value of the tuple
    data, _, _, _ = dataset.get_data(dataset_format='dataframe')
    return data


def load_dataset(dataset_id: int, dataset_path: str) -> pd.DataFrame:
    file_path = Path(dataset_path)

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = get_dataset_from_openml(dataset_id)
        data.to_csv(file_path)

    return pd.read_csv(file_path)


def get_train_test_data(df: pd.DataFrame, target: str, split_size):
    x = df.drop(columns=[target])
    y = df[target]
    return train_test_split(x, y, test_size=split_size)


def timer(func):
    """
    A decorator to measure and print the execution time of a function.

    Args:
    - func (function): The function to be wrapped by the timer decorator.

    Returns:
    - wrapper (function): A wrapped function that calculates and prints the time
                           taken to execute the original function.

    This decorator can be used to wrap functions and output their execution time
    in seconds.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{func.__name__} executed in {duration:.4f} seconds\n")
        return result

    return wrapper


def get_rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def convert_to_numpy(*args: 'pd.DataFrame | np.ndarray | pd.Series') -> tuple[np.ndarray, ...]:
    return tuple(arg.to_numpy() if isinstance(arg, (pd.DataFrame, pd.Series)) else arg for arg in args)


def train_test_to_numpy(x_train, x_test, y_train, y_test) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def get_regression_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return np.std(y_true, ddof=1), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)
