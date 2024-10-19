import os
import pandas as pd
import arff

from assignment1.visualize import get_numeric_cols, get_histograms_for_numeric_data

if __name__ == '__main__':
    WORKING_DIR = os.getcwd()
    PHISHING_RAW_DATA_PATH: str = WORKING_DIR + "\\data\\PhiUSIIL_Phishing_URL_Dataset.csv"
    SALARY_RAW_DATA_PATH: str = WORKING_DIR + "\\data\\dataset.arff"

    with open(SALARY_RAW_DATA_PATH, 'r') as f:
        dataset = arff.load(f)

    df_salary_raw_data = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    df_phishing_email_raw_data = pd.read_csv(PHISHING_RAW_DATA_PATH)

    dfs_to_visualize = {
        "salary_data": df_salary_raw_data,
        "phishing_url": df_phishing_email_raw_data
    }

    for dataset, df in zip(dfs_to_visualize.keys(), dfs_to_visualize.values()):
        get_histograms_for_numeric_data(
            working_dir=WORKING_DIR,
            df=df,
            num_cols=get_numeric_cols(df),
            dataset_name=dataset
        )
