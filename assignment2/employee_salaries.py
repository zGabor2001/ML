from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from assignment2.model import generate_hyperparameter_permutations, run_random_forest_with_varied_params, \
    ScratchRandomForest as SelfMadeRandomForest
from assignment2.util.data_utils import load_dataset

_DATASET_ID = 42125
_DATASET_PATH = 'data/employee_salaries.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'current_annual_salary'
_CORRELATION_DROP_THRESHOLD = 1.0

_OUTPUT_FOLDER = Path('output/employee_salaries')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'


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
    x = df.drop(columns=[_TARGET_VARIABLE])
    y = df[_TARGET_VARIABLE]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=_TEST_SPLIT_SIZE)

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


if __name__ == '__main__':
    explore_employee_salaries_dataset()
