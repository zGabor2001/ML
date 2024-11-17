import pandas as pd

from assignment1.utils import (impute_missing_values,
                               drop_multicorr_variables_form_df,
                               parallel_encoding,
                               optimize_data_types)
from assignment1.data_prep import CheckDatasetCondition
from assignment1.classifiers import SupportVectorMachineClassifier


def process_phishing_data(file_path: str, run_config: dict) -> pd.DataFrame:
    df_phishing: pd.DataFrame = pd.read_csv(file_path)

    df_phishing = df_phishing.sample(n=1000)

    phishing_dataset_condition = CheckDatasetCondition(df_phishing, run_config['phishing_target'])
    phishing_feature_results, phishing_target_results = phishing_dataset_condition.get_dataset_condition()

    df_phishing.drop(columns='FILENAME', inplace=True)
    df_phishing: pd.DataFrame = impute_missing_values(df_phishing)
    df_phishing: pd.DataFrame = parallel_encoding(df_phishing)
    if run_config['dtype_opt']:
        df_phishing: pd.DataFrame = optimize_data_types(df_phishing)
    #df_phishing: pd.DataFrame = drop_multicorr_variables_form_df(df_phishing, run_config, phishing_feature_results['multicollinearity'])

    return df_phishing


def process_road_safety_data(file_path: str, run_config: dict) -> pd.DataFrame:
    df_safety: pd.DataFrame = pd.read_csv(file_path, index_col=0)
    df_safety = df_safety.groupby(run_config['road_safety_target']).apply(lambda x: x.sample(n=1000, random_state=42))
    df_safety = df_safety.reset_index(drop=True)
    # not_needed_cols = ['Accident_Index', 'Vehicle_Reference_df_res']
    # df_safety.drop(not_needed_cols, inplace=True)

    # safety_dataset_condition = CheckDatasetCondition(df_safety, run_config['road_safety_target'])
    # safety_feature_results, safety_target_results = safety_dataset_condition.get_dataset_condition()
    if run_config['impute_missing']:
        df_safety: pd.DataFrame = impute_missing_values(df_safety)
    df_safety: pd.DataFrame = parallel_encoding(df_safety)
    if run_config['dtype_opt']:
        df_safety: pd.DataFrame = optimize_data_types(df_safety)
    # df_safety: pd.DataFrame = drop_multicorr_variables_form_df(df_safety, run_config, safety_feature_results['multicollinearity'])

    return df_safety


def fit_svm_model(df: pd.DataFrame, target: str) -> dict:
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
    svm = SupportVectorMachineClassifier(kernel='linear', C=1.0)

    x_train, x_test, y_train, y_test = svm.preprocess_data(df, target)
    svm.fit(x_train, y_train)
    results = svm.evaluate(x_test, y_test)

    return results
