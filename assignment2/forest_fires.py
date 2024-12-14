from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from assignment2.model import generate_hyperparameter_permutations, run_random_forest_with_varied_params, \
    RandomForest as SelfMadeRandomForest
from assignment2.preprocessing import periodic_spline_transformer

_DATASET_PATH = 'data/forestfires.csv'
_TARGET_VARIABLE = 'area'
_TEST_SPLIT_SIZE = 0.2

_OUTPUT_FOLDER = Path('output/forest_fires')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def explore_forest_fires_dataset():
    # Pandas DataFrame output for sklearn transformers
    set_config(transform_output='pandas')

    df = load_dataset(_DATASET_PATH)

    # data split into features and target variable
    # as well as into training and testing sets
    x = df.drop(columns=[_TARGET_VARIABLE])
    y = df[_TARGET_VARIABLE]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=_TEST_SPLIT_SIZE)

    # setup preprocessing pipeline
    # we use ordinal encoding and spline transformation for month and day
    # no missing values to impute this time
    preprocessing_pipeline = Pipeline([
        ('column transformations', ColumnTransformer([
            ('month', Pipeline([
                ('ordinal', OrdinalEncoder(
                    categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']])),
                ('spline', periodic_spline_transformer(period=12, n_splines=6))
            ]), ['month']),
            ('weekday', Pipeline([
                ('ordinal', OrdinalEncoder(categories=[['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']])),
                ('spline', periodic_spline_transformer(period=7, n_splines=3))
            ]), ['day'])
        ], remainder='passthrough', verbose_feature_names_out=False))
    ])

    # preprocess training and testing data (prevent data leakage by preprocessing them separately)
    x_train_transformed = preprocessing_pipeline.fit_transform(x_train)
    x_test_transformed = preprocessing_pipeline.transform(x_test)

    # run model with permutation of different hyperparameters
    params = generate_hyperparameter_permutations(
        no_of_trees=[50, 100, 200],
        max_depth=[20, 50, 500],
        min_samples=[10, 100, 200],
        feature_subset_size=[6, 12, 19],
    )

    results = run_random_forest_with_varied_params(
        model_cls=SelfMadeRandomForest,
        x_train=x_train_transformed,
        x_test=x_test_transformed,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=params,
        n_jobs=1,
        verbose=True
    )

    # save results
    _OUTPUT_HYPERPARAMETERS_FOLDER.mkdir(parents=True, exist_ok=True)
    results.to_csv(_OUTPUT_HYPERPARAMETERS_RESULTS)


if __name__ == '__main__':
    explore_forest_fires_dataset()
