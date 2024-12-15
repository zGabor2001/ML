from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from assignment2.model import generate_hyperparameter_permutations, run_random_forest_with_varied_params, \
    ScratchRandomForest as SelfMadeRandomForest, generate_knn_hyperparameter_permutations
from assignment2.model.runner import run_sklearn_model_with_varied_params
from assignment2.util.data_utils import load_dataset, get_train_test_data

_DATASET_ID = 42125
_DATASET_PATH = 'data/employee_salaries.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'current_annual_salary'
_CORRELATION_DROP_THRESHOLD = 1.0

_OUTPUT_FOLDER = Path('output/employee_salaries')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

_OUTPUT_KNN = _OUTPUT_FOLDER / 'knn'
_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS = _OUTPUT_KNN / 'parameter_permutations.csv'

_TEST_RUN=True

def explore_employee_salaries_dataset():
    # Pandas DataFrame output for sklearn transformers
    set_config(transform_output='pandas')

    df = load_dataset(_DATASET_ID, _DATASET_PATH)

    # split date_first_hired into year, month, day
    date_first_hired = pd.to_datetime(df['date_first_hired'])
    df['year_first_hired'] = date_first_hired.dt.year
    df['month_first_hired'] = date_first_hired.dt.month
    df['day_first_hired'] = date_first_hired.dt.day
    df.drop(columns=['date_first_hired', 'full_name'], inplace=True)

    # data split into features and target variable
    # as well as into training and testing sets
    x_train, x_test, y_train, y_test = get_train_test_data(df=df, target=_TARGET_VARIABLE, split_size=_TEST_SPLIT_SIZE)

    # setup larger preprocessing pipeline
    # Columns with missing values:
    # ---------------------------
    # gender
    # 2016_gross_pay_received
    # 2016_overtime_pay
    # underfilled_job_title

    gender_preprocessing_pipeline = Pipeline([
        ('constant_imputed', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('one_hot_encoded', OneHotEncoder(sparse_output=False))
    ])

    job_title_preprocessing_pipeline = Pipeline([
        ('constant_imputed', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('ordinal_encoded', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Gender variable is one-hot-encoded
    # Other categorical variables are ordinal encoded due to high cardinality
    # constant imputation for categorical variables
    # median imputation for numerical variables
    #
    # scaling / value transformations should be unnecessary for tree-based models
    preprocessing_pipeline = Pipeline([
        ('column transformations', ColumnTransformer([
            ('gender', gender_preprocessing_pipeline, ['gender']),
            ('job', job_title_preprocessing_pipeline, ['underfilled_job_title']),
            # Other transformations
            ('median_imputed', SimpleImputer(strategy='median'), ['2016_gross_pay_received', '2016_overtime_pay']),
            ('ordinal_encoded', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
             ['department', 'department_name', 'division', 'assignment_category', 'employee_position_title'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        )),
    ])

    # preprocess training and testing data (prevent data leakage by preprocessing them separately)
    x_train_transformed = preprocessing_pipeline.fit_transform(x_train)
    x_test_transformed = preprocessing_pipeline.transform(x_test)

    # run model with permutation of different hyperparameters
    if _TEST_RUN:
        params = generate_hyperparameter_permutations(
            no_of_trees=[50],
            max_depth=[20],
            min_samples=[10],
            feature_subset_size=[4],
        )
    else:
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

    # knn - unlike regression trees - requires scaling
    # hence we need a separate preprocessing pipeline
    preprocessing_pipeline_knn = Pipeline([
        ('column transformations', ColumnTransformer([
            ('gender', gender_preprocessing_pipeline, ['gender']),
            ('job', job_title_preprocessing_pipeline, ['underfilled_job_title']),
            # Other transformations
            ('median_imputed', SimpleImputer(strategy='median'), ['2016_gross_pay_received', '2016_overtime_pay']),
            ('ordinal_encoded', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
             ['department', 'department_name', 'division', 'assignment_category', 'employee_position_title'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        )),
        ('scaling', StandardScaler())
    ])

    x_train_transformed_knn = preprocessing_pipeline_knn.fit_transform(x_train)
    x_test_transformed_knn = preprocessing_pipeline_knn.transform(x_test)

    if _TEST_RUN:
        knn_params = generate_knn_hyperparameter_permutations(
            n_neighbors=[5],
            weights=['uniform'],
            leaf_size=[10]
        )
    else:
        knn_params = generate_knn_hyperparameter_permutations(
            n_neighbors=[5, 10, 20],
            weights=['uniform', 'distance'],
            leaf_size=[10, 30, 50]
        )

    knn_results = run_sklearn_model_with_varied_params(
        model_cls=KNeighborsRegressor,
        x_train=x_train_transformed_knn,
        x_test=x_test_transformed_knn,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=knn_params,
        n_jobs=1,
        verbose=True
    )

    _OUTPUT_KNN.mkdir(parents=True, exist_ok=True)
    knn_results.to_csv(_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS)

if __name__ == '__main__':
    explore_employee_salaries_dataset()
