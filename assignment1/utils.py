import pandas as pd


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.Series: Series containing the count of missing values for each column.
    """
    missing_counts = df.isnull().sum()
    total_counts = df.shape[0]
    result = pd.DataFrame({
        'Missing Values': missing_counts,
        'Total Entries': total_counts,
        'Percentage Missing': (missing_counts / total_counts) * 100
    })
    return result


def get_data_types_for_df_columns(df: pd.DataFrame) -> pd.Series:
    """
        Get the data types of each column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.Series: Series containing the data types of each column.
        """
    return df.dtypes


def check_feature_scaling(df):
    """
    Check the range (min, max), mean, and standard deviation for numerical features in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with min, max, mean, and std for each numerical column.
    """
    # Select numerical columns only
    numerical_columns = df.select_dtypes(include=['number'])

    # Calculate min, max, mean, and standard deviation for each numerical feature
    scaling_stats = numerical_columns.describe().T[['min', 'max', 'mean', 'std']]

    return scaling_stats
