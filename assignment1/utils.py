import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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


def check_feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the range (min, max), mean, and standard deviation for numerical features in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with min, max, mean, and std for each numerical column.
    """
    numerical_columns: pd.DataFrame = df.select_dtypes(include=['number'])
    scaling_stats: pd.DataFrame = numerical_columns.describe().T[['min', 'max', 'mean', 'std']]
    return scaling_stats


def detect_outliers_with_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in numerical features using the IQR method.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numerical features.

    Returns:
    - pd.DataFrame: DataFrame with outlier status (True/False) for each value in the DataFrame.
    """
    numerical_columns = df.select_dtypes(include=['number'])
    outliers_df = pd.DataFrame(index=df.index)
    for col in numerical_columns.columns:
        q1 = numerical_columns[col].quantile(0.25)
        q3 = numerical_columns[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_df[col] = (numerical_columns[col] < lower_bound) | (numerical_columns[col] > upper_bound)

    return outliers_df


def check_categorical_variables(df: pd.DataFrame) -> list:
    """
    Check for categorical variables in the DataFrame by identifying columns with
    categorical or object data types.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - list: A list of column names that are categorical.
    """
    categorical_columns: list = df.select_dtypes(include=['object', 'category']).columns.tolist()

    encoded_categorical_columns = []
    for col in df.select_dtypes(include=['number']).columns:
        unique_values: int = df[col].nunique()
        if unique_values < 10:
            encoded_categorical_columns.append(col)

    categorical_columns: list = categorical_columns + encoded_categorical_columns

    return categorical_columns


def check_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix and plot a heatmap to visualize high correlations.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numerical features.

    Returns:
    - pd.DataFrame: The correlation matrix.
    """
    df_num: pd.DataFrame = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr()

    # Plot the heatmap
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    # plt.title("Correlation Matrix Heatmap")
    # plt.show()

    return corr_matrix


def check_vif(df):
    """
    Calculate Variance Inflation Factor (VIF) for each feature to check for multicollinearity.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numerical features.

    Returns:
    - pd.DataFrame: DataFrame with VIF values for each feature.
    """
    # Add a constant column for the intercept term (required for VIF calculation)
    df_with_const = add_constant(df)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]

    return vif_data


def check_class_balance(df, target_var):
    """
    Check the distribution of the target variable to identify class imbalance.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_var (str): The name of the target variable/column.

    Returns:
    - pd.Series: The count of occurrences for each class in the target variable.
    """
    class_distribution = df[target_var].value_counts()

    # Plot the class distribution
    # plt.figure(figsize=(8, 6))
    # sns.countplot(data=df, x=target_var, palette='Set2')
    # plt.title(f"Class Distribution for {target_var}")
    # plt.ylabel('Frequency')
    # plt.xlabel('Class')
    # plt.show()

    return class_distribution
