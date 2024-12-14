import pandas as pd
from sklearn.impute import SimpleImputer


def columns_with_missing_values(df: pd.DataFrame) -> list[str]:
    """
    Get the columns in the dataframe with missing values

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to check

    Returns
    -------
    list
        The list of columns with missing values
    """
    return df.columns[df.isnull().any()].tolist()


def impute_missing_values(
        df: pd.DataFrame,
        columns: list[str],
        strategy: str = 'mean',
        constant: str = None) -> pd.DataFrame:
    """
    Impute missing values in the dataframe
    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be imputed.
        columns (list[str]): A list of column names where missing values need to be imputed.
        strategy (str): The imputation strategy to use. Options include:
            - 'mean': Replace missing values with the mean of the column (default).
            - 'median': Replace missing values with the median of the column.
            - 'most_frequent': Replace missing values with the most frequent value in the column.
            - 'constant': Replace missing values with a constant value
        constant (str): The constant value to use when strategy is 'constant'. Ignored for other strategies.

    Returns:
        pd.DataFrame: A copy of the DataFrame with missing values in the specified columns imputed.

    Raises:
        ValueError: If an invalid strategy is provided or if constant is not provided when strategy is 'constant'.
    """
    if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
        raise ValueError(f"Invalid imputation strategy: {strategy}")

    if strategy == 'constant' and constant is None:
        raise ValueError("constant value must be provided when strategy is 'constant'")

    imputer = SimpleImputer(strategy=strategy, fill_value=constant)
    modified_df = df.copy()
    modified_df[columns] = imputer.fit_transform(modified_df[columns])
    return modified_df

