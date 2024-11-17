import pandas as pd
import os

from assignment1.utils import (impute_missing_values,
                               drop_multicorr_variables_form_df,
                               parallel_encoding,
                               optimize_data_types,
                               timer,)
from assignment1.data_prep import CheckDatasetCondition
from assignment1.classifiers import SupportVectorMachineClassifier


@timer
def process_phishing_data(run_config: dict) -> pd.DataFrame:
    file_path = run_config['phishing_data_path']
    df_phishing: pd.DataFrame = pd.read_csv(file_path)
    df_phishing = df_phishing.drop(columns='FILENAME')

    sample_size = run_config['phishing_sample_size']
    if sample_size is not None:
        df_phishing = df_phishing.groupby(run_config['phishing_target']).apply(
            lambda x: x.sample(n=sample_size, random_state=42))
        df_phishing = df_phishing.reset_index(drop=True)

    if run_config['dataset_report']:
        phishing_dataset_condition = CheckDatasetCondition(
            df_phishing, run_config['phishing_target'])
        phishing_feature_results, phishing_target_results = phishing_dataset_condition.get_dataset_condition()

    if run_config['impute_missing']:
        df_phishing: pd.DataFrame = impute_missing_values(df_phishing)

    df_phishing: pd.DataFrame = parallel_encoding(df_phishing)

    if run_config['dtype_opt']:
        df_phishing: pd.DataFrame = optimize_data_types(df_phishing)

    # df_phishing: pd.DataFrame = drop_multicorr_variables_form_df(df_phishing, run_config, phishing_feature_results['multicollinearity'])
    df_phishing.to_csv(os.path.join(
        run_config['data_path'], 'prep_phishing_data.csv'), encoding='utf-8')

    return df_phishing


@timer
def process_road_safety_data(run_config: dict) -> pd.DataFrame:
    file_path = run_config['road_safety_data_path']
    df_safety: pd.DataFrame = pd.read_csv(file_path, index_col=0)
    df_safety = df_safety.drop(
        columns=['Accident_Index', 'Vehicle_Reference_df_res'])

    sample_size = run_config['phishing_sample_size']
    if sample_size is not None:
        df_safety = df_safety.groupby(run_config['road_safety_target']).apply(
            lambda x: x.sample(n=sample_size, random_state=42))
        df_safety = df_safety.reset_index(drop=True)

    if run_config['dataset_report']:
        safety_dataset_condition = CheckDatasetCondition(
            df_safety, run_config['road_safety_target'])
        safety_feature_results, safety_target_results = safety_dataset_condition.get_dataset_condition()

    if run_config['impute_missing']:
        df_safety: pd.DataFrame = impute_missing_values(df_safety)
    df_safety: pd.DataFrame = parallel_encoding(df_safety)

    if run_config['dtype_opt']:
        df_safety: pd.DataFrame = optimize_data_types(df_safety)

    # df_safety: pd.DataFrame = drop_multicorr_variables_form_df(df_safety, run_config, safety_feature_results['multicollinearity'])
    df_safety.to_csv(os.path.join(
        run_config['data_path'], 'prep_road_data.csv'), encoding='utf-8')

    return df_safety


@timer
def fit_svm_model(df: pd.DataFrame, target: str, kernel: str, c: float) -> dict:
    """
    Train and evaluate an SVM model using the provided DataFrame and target column.

    Parameters:
    - df: pd.DataFrame
        Input DataFrame containing features and target column.
    - target: str
        The name of the target column.

    Returns:
    - dict: Evaluation results containing accuracy and a classification report.
    """
    svm = SupportVectorMachineClassifier(kernel=kernel, C=c)

    x_train, x_test, y_train, y_test = svm.preprocess_data(df, target)
    svm.fit(x_train, y_train)
    results = svm.evaluate(x_test, y_test)

    return results
