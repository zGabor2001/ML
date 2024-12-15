from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from assignment2.model import generate_hyperparameter_permutations, ScratchRandomForest as SelfMadeRandomForest, generate_knn_hyperparameter_permutations
from assignment2.model.llm_random_forest import LLMRandomForestRegressor
from assignment2.util.data_utils import timer

from assignment2.model.runner import run_sklearn_model_with_varied_params, train_all_random_forests_on_data
from assignment2.util.data_utils import load_dataset, get_train_test_data

_DATASET_ID = 43723
_DATASET_PATH = 'data/toronto_rental.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'Price'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = False
_RANDOM_FOREST_CLASSES_FOR_TRAINING = [RandomForestRegressor,
                                       SelfMadeRandomForest, LLMRandomForestRegressor]

_OUTPUT_FOLDER = Path('output/toronto_rental')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

_OUTPUT_KNN = _OUTPUT_FOLDER / 'knn'
_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS = _OUTPUT_KNN / 'parameter_permutations.csv'

def prepare_toronto_rental_dataset():
    df = load_dataset(_DATASET_ID, _DATASET_PATH)
    df = df.iloc[:, 1:]
    df['Price'] = df['Price'].str.replace(
        ',', '').astype(float)    # Is this a good idea???
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

    x_train_transformed_rf = preprocessing_pipeline.fit_transform(x_train)
    x_test_transformed_rf = preprocessing_pipeline.transform(x_test)


    # knn - unlike regression trees - requires scaling
    # hence we need a separate preprocessing pipeline
    preprocessing_pipeline_knn = Pipeline([
        ('column transformations', ColumnTransformer([
            ('address', address_preprocessing_pipeline, ['Address'])
        ], remainder='passthrough', verbose_feature_names_out=False)),
        ('scaling', StandardScaler())
    ])

    x_train_transformed_knn = preprocessing_pipeline_knn.fit_transform(x_train)
    x_test_transformed_knn = preprocessing_pipeline_knn.transform(x_test)

    return x_train_transformed_rf, x_test_transformed_rf, x_train_transformed_knn, x_test_transformed_knn, y_train, y_test


@timer
def explore_toronto_rental_dataset():
    (x_train_transformed_rf,
     x_test_transformed_rf,
     x_train_transformed_knn,
     x_test_transformed_knn,
     y_train,
     y_test) = prepare_toronto_rental_dataset()

    # run model with permutation of different hyperparameters
    if _TEST_RUN:
        rf_params = generate_hyperparameter_permutations(
            no_of_trees=[50],
            max_depth=[20],
            min_samples=[10],
            feature_subset_size=[2],
        )
        knn_params = generate_knn_hyperparameter_permutations(
            n_neighbors=[5],
            weights=['uniform'],
            leaf_size=[10]
        )
    else:
        rf_params = generate_hyperparameter_permutations(
            no_of_trees=[50, 70, 100],
            max_depth=[20, 50, 70],
            min_samples=[10, 100, 200],
            feature_subset_size=[2, 3, 5],
        )
        knn_params = generate_knn_hyperparameter_permutations(
            n_neighbors=[5, 10, 20],
            weights=['uniform', 'distance'],
            leaf_size=[10, 30, 50]
        )

    train_all_random_forests_on_data(random_forests=_RANDOM_FOREST_CLASSES_FOR_TRAINING,
                                     params=rf_params,
                                     x_train_transformed=x_train_transformed_rf,
                                     x_test_transformed=x_test_transformed_rf,
                                     y_train=y_train,
                                     y_test=y_test,
                                     output_folder=_OUTPUT_HYPERPARAMETERS_FOLDER)

    knn_results = run_sklearn_model_with_varied_params(
        model_cls=KNeighborsRegressor,
        x_train=x_train_transformed_knn,
        x_test=x_test_transformed_knn,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=knn_params,
        verbose=True,

    )

    # save knn results
    _OUTPUT_KNN.mkdir(parents=True, exist_ok=True)
    knn_results.to_csv(_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS)


if __name__ == "__main__":
    explore_toronto_rental_dataset()
