from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

from assignment2.model import generate_hyperparameter_permutations, run_random_forest_with_varied_params, \
    ScratchRandomForest as SelfMadeRandomForest
from assignment2.util.data_utils import load_dataset, get_train_test_data

_DATASET_ID = 43723
_DATASET_PATH = 'data/toronto_rental.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'Price'
_CORRELATION_DROP_THRESHOLD = 1.0

_OUTPUT_FOLDER = Path('output/toronto_rental')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'


def explore_toronto_rental_dataset():
    df = load_dataset(_DATASET_ID, _DATASET_PATH)
    df = df.iloc[:, 1:]
    x_train, x_test, y_train, y_test = get_train_test_data(df=df, target=_TARGET_VARIABLE, split_size=_TEST_SPLIT_SIZE)

    df['Price'] = df['Price'].str.replace(',', '').astype(float)

    address_preprocessing_pipeline = Pipeline([
        ('ordinal_encoded', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessing_pipeline = Pipeline([
        ('column transformations', ColumnTransformer([
            ('address', address_preprocessing_pipeline, ['Address'])
        ]))
    ])

    x_train_transformed = preprocessing_pipeline.fit_transform(x_train)
    x_test_transformed = preprocessing_pipeline.transform(x_test)

    # run model with permutation of different hyperparameters
    params = generate_hyperparameter_permutations(
        no_of_trees=[50, 100, 200],
        max_depth=[20, 50, 500],
        min_samples=[10, 100, 200],
        feature_subset_size=[4, 9, 14],
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


if __name__ == "__main__":
    explore_toronto_rental_dataset()
