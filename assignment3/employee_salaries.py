from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from assignment2.model import ScratchRandomForest as SelfMadeRandomForest
from assignment2.model.llm_random_forest import LLMRandomForestRegressor
from assignment2.util.data_utils import load_dataset, get_train_test_data, timer, convert_to_numpy

_DATASET_ID = 42125
_DATASET_PATH = 'data/employee_salaries.csv'
_TEST_SPLIT_SIZE = 0.2
_TARGET_VARIABLE = 'current_annual_salary'
_CORRELATION_DROP_THRESHOLD = 1.0
_TEST_RUN = True
_RANDOM_FOREST_CLASSES_FOR_TRAINING = [RandomForestRegressor,
                                       SelfMadeRandomForest, LLMRandomForestRegressor]

_OUTPUT_FOLDER = Path('output/employee_salaries')
_OUTPUT_HYPERPARAMETERS_FOLDER = _OUTPUT_FOLDER / 'parameter_permutation'
_OUTPUT_HYPERPARAMETERS_RESULTS = _OUTPUT_HYPERPARAMETERS_FOLDER / 'results.csv'


@timer
def prepare_employee_salaries_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_dataset(_DATASET_ID, _DATASET_PATH)
    df = df.iloc[:100, :]
    # split date_first_hired into year, month, day
    date_first_hired = pd.to_datetime(df['date_first_hired'])
    df['year_first_hired'] = date_first_hired.dt.year
    df['month_first_hired'] = date_first_hired.dt.month
    df['day_first_hired'] = date_first_hired.dt.day
    df.drop(columns=['date_first_hired', 'full_name'], inplace=True)

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

    x_train = preprocessing_pipeline.fit_transform(x_train)
    x_test = preprocessing_pipeline.transform(x_test)

    x_train, x_test, y_train, y_test = convert_to_numpy(x_train, x_test, y_train, y_test)

    return x_train, x_test, y_train, y_test
