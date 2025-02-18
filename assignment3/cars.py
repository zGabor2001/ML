from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from assignment3.util.data_utils import load_dataset, get_train_test_data, timer

_DATASET_ID = 44994
_DATASET_PATH = 'data/cars.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'Price'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = False


_OUTPUT_FOLDER = Path('output/cars')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

_OUTPUT_KNN = _OUTPUT_FOLDER / 'knn'
_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS = _OUTPUT_KNN / 'parameter_permutations.csv'


def prepare_cars_dataset():
    df = load_dataset(_DATASET_ID, _DATASET_PATH)

    df = df.iloc[:100, :]

    X = df.drop(columns=[_TARGET_VARIABLE])

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # data split into features and target variable
    # as well as into training and testing sets
    x_train, x_test, y_train, y_test = get_train_test_data(df=df, target=_TARGET_VARIABLE, split_size=_TEST_SPLIT_SIZE)
    # Apply preprocessing to the training and testing data
    x_train = pipeline.fit_transform(x_train)
    x_test = pipeline.transform(x_test)
    return x_train, x_test, y_train, y_test
