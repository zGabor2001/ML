import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Label encode the columns in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to encode
    columns : list
        The list of columns to encode

    Returns
    -------
    pandas.DataFrame
        The dataframe with the columns encoded
    """
    modified_df = df.copy()
    for column in columns:
        label_encoder = LabelEncoder()
        modified_df[column] = label_encoder.fit_transform(modified_df[column])
    return modified_df


def one_hot_encode_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    One hot encode the columns in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to encode
    columns : list
        The list of columns to encode

    Returns
    -------
    pandas.DataFrame
        The dataframe with the columns encoded
    """
    # get_dummies operates on a new dataframe, so no need to copy
    return pd.get_dummies(df, columns=columns)
