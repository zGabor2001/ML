import openml
import pandas as pd


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

