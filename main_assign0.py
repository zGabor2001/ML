import os
import pandas as pd

from assignment0.visualize import get_numeric_cols, get_histograms_for_numeric_data

if __name__ == '__main__':
    WORKING_DIR = os.getcwd()
    PHISHING_RAW_DATA_PATH: str = WORKING_DIR + "/assignment1/data/PhiUSIIL_Phishing_URL_Dataset.csv"
    SALARY_RAW_DATA_PATH: str = WORKING_DIR + "/assignment1/data/road_safety.csv"


    df_phishing_email_raw_data = pd.read_csv(PHISHING_RAW_DATA_PATH)
    df_salary_raw_data = pd.read_csv(SALARY_RAW_DATA_PATH)

    dfs_to_visualize = {
        "salary_data": df_salary_raw_data,
        #"phishing_url": df_phishing_email_raw_data
    }

    import matplotlib.pyplot as plt

    freq_counts = df_phishing_email_raw_data['label'].value_counts()  # Count unique values

    plt.figure(figsize=(8, 5))
    freq_counts.plot(kind='bar', color='teal')  # Create a bar plot
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Classes in label')
    plt.xticks(rotation=0)
    plt.show()

    # for dataset, df in zip(dfs_to_visualize.keys(), dfs_to_visualize.values()):
    #     get_histograms_for_numeric_data(
    #         working_dir=WORKING_DIR,
    #         df=df,
    #         num_cols=get_numeric_cols(df),
    #         dataset_name=dataset
    #     )

