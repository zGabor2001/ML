from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

from assignment3.util.data_utils import load_dataset, get_train_test_data, timer, train_test_to_numpy, convert_to_numpy

_DATASET_ID = 43723
_DATASET_PATH = 'data/toronto_rental.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'Price'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = False

_OUTPUT_FOLDER = Path('output/toronto_rental')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

_OUTPUT_KNN = _OUTPUT_FOLDER / 'knn'
_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS = _OUTPUT_KNN / 'parameter_permutations.csv'


@timer
def prepare_toronto_rental_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_dataset(_DATASET_ID, _DATASET_PATH)

    if _TEST_RUN:
        df = df.iloc[:100, :]
    print("Dimensions for training:", df.shape)

    df = df.iloc[:, 1:]
    df['Price'] = df['Price'].str.replace(
        ',', '').astype(float)
    x_train, x_test, y_train, y_test = get_train_test_data(
        df=df, target=_TARGET_VARIABLE, split_size=_TEST_SPLIT_SIZE)

    address_preprocessing_pipeline = Pipeline([
        ('ordinal_encoded', OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessing_pipeline = Pipeline([
        ('column transformations', ColumnTransformer([
            ('address', address_preprocessing_pipeline, ['Address'])
        ], remainder='passthrough', verbose_feature_names_out=False))
    ])

    x_train = preprocessing_pipeline.fit_transform(x_train)
    x_test = preprocessing_pipeline.transform(x_test)

    x_train, x_test, y_train, y_test = convert_to_numpy(x_train, x_test, y_train, y_test)

    return x_train, x_test, y_train, y_test
