from typing import Dict

import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from functools import wraps
import warnings
import dask.dataframe as dd
import models

RANDOM_STATE = 42
BREAST_SAMPLE_SIZE = 10
LOAN_SAMPLE_SIZE = 10
PHISHING_DATA_SAMPLE = 10
ROAD_SAFETY_SAMPLE_SIZE = 10
PHISHING_TARGET = 'label'
ROAD_SAFETY_TARGET = 'Casualty_Severity'

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

df_breast_cancer = pd.read_csv('./data/breast-cancer-diagnostic.shuf.lrn.csv')
df_road_safety = pd.read_csv('./data/road_safety.csv')
df_phishing = pd.read_csv('./data/PhiUSIIL_Phishing_URL_Dataset.csv')
df_loan = pd.read_csv('./data/loan-10k.lrn.csv')

df_dict = {'breast_cancer': df_breast_cancer,
           'phishing': df_phishing,
           'road_safety': df_road_safety,
           'loan': df_loan
           }

for key in df_dict.keys():
    print(f'{key} missing values: {df_dict[key].isnull().sum().any()}')


def get_metrics_dict(
        accuracy: float,
        f1: float,
        precision: float,
        recall: float,
) -> Dict[str, float]:
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


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
        print(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper


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
def parallel_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform parallel label encoding on categorical columns in a DataFrame using Dask.

    Args:
    - df (pd.DataFrame): The input DataFrame containing categorical data.

    Returns:
    - pd.DataFrame: The DataFrame with label-encoded categorical columns.

    This function converts categorical columns in the DataFrame to numerical codes using label encoding
    while utilizing Dask to perform the operation in parallel across multiple partitions,
    improving performance for large datasets.
    """
    ddf = dd.from_pandas(df, npartitions=4)
    le = LabelEncoder()
    ddf_encoded = ddf.map_partitions(label_encode, le)
    df_encoded = ddf_encoded.compute()
    return df_encoded


@timer
def label_encode(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    """
    Label encode the categorical columns in a DataFrame using the provided LabelEncoder.

    Args:
    - df (pd.DataFrame): The input DataFrame containing categorical columns.
    - le (LabelEncoder): A fitted LabelEncoder instance.

    Returns:
    - pd.DataFrame: The DataFrame with label-encoded categorical columns.

    This function applies label encoding to all categorical columns (non-numeric) in the DataFrame using
    a provided LabelEncoder object. It converts each category into a corresponding numeric label.
    """
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = le.fit_transform(df[column].astype(str))
    return df


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
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
        elif pd.api.types.is_numeric_dtype(col_type):
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')

    return df


def sample_dataset(df, sample_size, target):
    if sample_size is not None:
        df = df.groupby(target).apply(
            lambda x: x.sample(n=sample_size, random_state=42))
        df = df.reset_index(drop=True)
    return df


def prep_breast_cancer_data(df):
    df = sample_dataset(df, BREAST_SAMPLE_SIZE, 'class')
    df['class'] = df['class'].astype(int)
    X = df.drop(columns=['ID', 'class'])
    Y = df['class']
    return X, Y


def prep_phishing_data(df):
    df = df.drop(columns='FILENAME')

    df = sample_dataset(df, PHISHING_DATA_SAMPLE, PHISHING_TARGET)

    df: pd.DataFrame = impute_missing_values(df)
    df: pd.DataFrame = parallel_encoding(df)
    df: pd.DataFrame = optimize_data_types(df)

    X = df.drop(columns=[PHISHING_TARGET])
    Y = df[PHISHING_TARGET]

    return X, Y


def prep_road_safety_data(df):
    df = df.drop(
        columns=['Accident_Index', 'Vehicle_Reference_df_res'])

    df = sample_dataset(df, ROAD_SAFETY_SAMPLE_SIZE, ROAD_SAFETY_TARGET)

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

    df: pd.DataFrame = impute_missing_values(df)
    df: pd.DataFrame = parallel_encoding(df)
    df: pd.DataFrame = optimize_data_types(df)

    return df.drop(columns=[ROAD_SAFETY_TARGET]), df[ROAD_SAFETY_TARGET]


def prep_loan_data(df):
    df = df.drop(columns=['ID'])
    df = sample_dataset(df, LOAN_SAMPLE_SIZE, 'grade')

    X = df.drop(columns=['grade'])

    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

    # standardize the numeric columns
    X[numeric_columns] = StandardScaler().fit_transform(X[numeric_columns])

    # one hot encode the categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns)

    Y = df['grade']

    return X, Y


model_results = []
for key in df_dict.keys():
    if key == 'breast_cancer':
        X, Y = prep_breast_cancer_data(df_dict[key])
    elif key == 'phishing':
        X, Y = prep_phishing_data(df_dict[key])
    elif key == 'road_safety':
        X, Y = prep_road_safety_data(df_dict[key])
    elif key == 'loan':
        X, Y = prep_loan_data(df_dict[key])
    else:
        raise KeyError('Invalid dataset key')
    model_results.append(models.run_models(X, Y, 42))

print(model_results)
