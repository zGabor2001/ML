import pandas as pd
import numpy as np
import dask.dataframe as dd
import time
from functools import wraps
import arff

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def process_arff(file_path: str):
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    df_safety: pd.DataFrame = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    df_safety.to_csv('C:\\DS\\repos\\ML\\assignment1\\data\\road_safety.csv', encoding='utf-8')


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        duration = end_time - start_time  # Calculate the duration
        print(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper


@timer
def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes the data types of the DataFrame's columns for memory efficiency.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be optimized.

    Returns:
    - pd.DataFrame: A DataFrame with optimized data types.
    """
    for col in df.columns:
        col_type = df[col].dtypes

        if col_type == 'object':
            # Convert to 'category' if the number of unique values is significantly less than the total count
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
        elif pd.api.types.is_numeric_dtype(col_type):
            # Downcast integers
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            # Downcast floats
            elif pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')

    return df


@timer
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


@timer
def get_data_types_for_df_columns(df: pd.DataFrame) -> pd.Series:
    """
        Get the data types of each column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.Series: Series containing the data types of each column.
        """
    return df.dtypes


@timer
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


@timer
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


@timer
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


@timer
def check_correlation_matrix(df: pd.DataFrame, threshold: float = 0.8) -> list:
    """
    Calculate the correlation matrix and plot a heatmap to visualize high correlations.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numerical features.

    Returns:
    - pd.DataFrame: The correlation matrix.
    """
    df_num: pd.DataFrame = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_var = [(column, row) for column in upper.columns for row in upper.index if
                     abs(upper.loc[row, column]) > threshold]

    return high_corr_var


@timer
def calculate_vif_for_columns(df, columns):
    df_subset = df[columns]
    df_with_const = add_constant(df_subset)

    vif_data = pd.DataFrame()
    vif_data['Variable'] = df_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]

    return vif_data


@timer
def select_variable_to_remove(df, pairs):
    variables_to_remove = []

    for pair in pairs:
        var1, var2 = pair

        vif_data = calculate_vif_for_columns(df, [var1, var2])
        vif_var1 = vif_data[vif_data['Variable'] == var1]['VIF'].values[0]
        vif_var2 = vif_data[vif_data['Variable'] == var2]['VIF'].values[0]

        if vif_var1 > vif_var2:
            variables_to_remove.append(var1)
        else:
            variables_to_remove.append(var2)

    return variables_to_remove


@timer
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame.
    Numerical columns are filled with the mean,
    while categorical columns are filled with the mode (most frequent value).

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with imputed values
    """
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Numerical columns
            df[column].fillna(df[column].mean(), inplace=True)
        else:  # Categorical columns
            df[column].fillna(df[column].mode()[0])
    return df


@timer
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

    return class_distribution


@timer
def get_variables_with_pca(df: pd.DataFrame, n_comp: int, threshold: float = 0.5) -> pd.DataFrame:
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_comp)
    pca.fit(df_scaled)
    loadings = pca.components_

    loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i + 1}' for i in range(n_comp)], index=df.columns)
    important_variables = loadings_df.apply(lambda x: np.abs(x) > threshold, axis=0)
    important_vars = important_variables.any(axis=1)

    return loadings_df.index[important_vars].tolist()


@timer
def pd_label_encode(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.Categorical(df[column]).codes
    return df


@timer
def parallel_encoding(df: pd.DataFrame) -> pd.DataFrame:
    ddf = dd.from_pandas(df, npartitions=4)
    le = LabelEncoder()
    ddf_encoded = ddf.map_partitions(label_encode, le)
    df_encoded = ddf_encoded.compute()
    return df_encoded


@timer
def label_encode(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = le.fit_transform(df[column].astype(str))
    return df


@timer
def drop_multicorr_variables_form_df(df: pd.DataFrame, config: dict, multicorr_cols: list):
    if config['remove_vars_multicorr']:
        variables_to_remove = list(set(select_variable_to_remove(df, multicorr_cols)))
    elif config['remove_vars_pca']:
        df_safety_numeric = df.select_dtypes(include=['float64', 'int64'])
        variables_to_remove = list(set(get_variables_with_pca(df_safety_numeric,
                                                              len(df.columns) - int(len(df.columns) / 2))
                                       )
                                   )
    else:
        return df
    df.drop(columns=list(set(variables_to_remove)), inplace=True)
    return df
