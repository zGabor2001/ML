import pandas as pd
import arff

from assignment1.utils import impute_missing_values
from assignment1.data_prep import CheckDatasetCondition
from assignment1.classifiers import SupportVectorMachineClassifier


def process_phishing_data(file_path: str) -> pd.DataFrame:
    df_phishing: pd.DataFrame = pd.read_csv(file_path)

    #phishing_dataset_condition = CheckDatasetCondition(df_phishing, ['label'])
    #phishing_feature_results, phishing_target_results = phishing_dataset_condition.get_dataset_condition()

    df_phishing = df_phishing.iloc[:5000, :]

    df_phishing = impute_missing_values(df_phishing)

    return df_phishing


def process_road_safety_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    df_safety: pd.DataFrame = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    df_safety = df_safety.iloc[:5000, :]

    #safety_dataset_condition = CheckDatasetCondition(df_safety, ['Casualty_Type'])
    #safety_feature_results, safety_target_results = safety_dataset_condition.get_dataset_condition()

    # Impute missing values
    df_safety = impute_missing_values(df_safety)

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
    # Initialize the SVM classifier
    svm = SupportVectorMachineClassifier(kernel='linear', C=1.0)

    # Preprocess the data
    x_train, x_test, y_train, y_test = svm.preprocess_data(df, target)

    # Fit the model
    svm.fit(x_train, y_train)

    # Evaluate the model on the test set
    results = svm.evaluate(x_test, y_test)

    return results
