from pathlib import Path
from typing import Tuple
import numpy as np

from assignment3.util.data_utils import load_dataset, get_train_test_data, timer, convert_to_numpy

_DATASET_ID = 43918
_DATASET_PATH = 'data/energy_efficiency.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'Y1'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = True


_OUTPUT_FOLDER = Path('output/energy_efficiency')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'


@timer
def prepare_energy_efficiency_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_dataset(_DATASET_ID, _DATASET_PATH)
    if _TEST_RUN:
        df = df.iloc[:100, :]

    x_train, x_test, y_train, y_test = get_train_test_data(df=df, target=_TARGET_VARIABLE, split_size=_TEST_SPLIT_SIZE)

    x_train, x_test, y_train, y_test = convert_to_numpy(x_train, x_test, y_train, y_test)

    return x_train, x_test, y_train, y_test
