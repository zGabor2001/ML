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

from assignment2.model import generate_hyperparameter_permutations, ScratchRandomForest as SelfMadeRandomForest, \
    generate_knn_hyperparameter_permutations
from assignment2.model.llm_random_forest import LLMRandomForestRegressor
from assignment2.model.runner import run_sklearn_model_with_varied_params
from assignment2.model.runner import train_all_random_forests_on_data
from assignment2.util.data_utils import load_dataset, get_train_test_data, timer

_DATASET_ID = 42125
_DATASET_PATH = 'data/employee_salaries.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'current_annual_salary'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = False
_RANDOM_FOREST_CLASSES_FOR_TRAINING = [RandomForestRegressor,
                                       SelfMadeRandomForest, LLMRandomForestRegressor]

_OUTPUT_FOLDER = Path('output/employee_salaries')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'

_OUTPUT_KNN = _OUTPUT_FOLDER / 'knn'
_OUTPUT_KNN_HYPERPARAMETER_PERMUTATIONS = _OUTPUT_KNN / 'parameter_permutations.csv'


def prepare_employee_salaries_dataset():
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
    x_train_transformed_rf = preprocessing_pipeline.fit_transform(x_train)
    x_test_transformed_rf = preprocessing_pipeline.transform(x_test)

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
    return x_train_transformed_rf, x_test_transformed_rf, x_train_transformed_knn, x_test_transformed_knn, y_train, y_test