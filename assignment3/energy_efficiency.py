from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from util.data_utils import load_dataset, get_train_test_data, timer

_DATASET_ID = 43918
_DATASET_PATH = 'data/energy_efficiency.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'Y1'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = False


_OUTPUT_FOLDER = Path('output/energy_efficiency')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

_OUTPUT_KNN = _OUTPUT_FOLDER / 'knn'
_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS = _OUTPUT_KNN / 'parameter_permutations.csv'


def prepare_energy_efficiency_dataset():
    df = load_dataset(_DATASET_ID, _DATASET_PATH)

    

    # data split into features and target variable
    # as well as into training and testing sets
    x_train, x_test, y_train, y_test = get_train_test_data(df=df, target=_TARGET_VARIABLE, split_size=_TEST_SPLIT_SIZE)

    
    return x_train, x_test, y_train, y_test
