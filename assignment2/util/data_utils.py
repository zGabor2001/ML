import openml
import pandas as pd
from pathlib import Path
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
        data = get_dataset_from_openml(dataset_id)
        data.to_csv(file_path)

    return pd.read_csv(file_path)


def get_train_test_data(df: pd.DataFrame, target: str, split_size):
    x = df.drop(columns=[target])
    y = df[target]
    return train_test_split(x, y, test_size=split_size)

