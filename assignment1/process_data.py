import pandas as pd
import arff


def process_phishing_data(file_path: str) -> pd.DataFrame:
    df_phishing: pd.DataFrame = pd.read_csv(file_path)

    return df_phishing


def process_road_safety_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    df_salary: pd.DataFrame = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    return df_salary
