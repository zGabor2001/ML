import os

from assignment1.process_data import *
from assignment1.road_safety_prep import CheckDatasetCondition

if __name__ == '__main__':
    WORKING_DIR: str = os.getcwd()
    PHISHING_RAW_DATA_PATH: str = WORKING_DIR + "\\assignment0\\data\\PhiUSIIL_Phishing_URL_Dataset.csv"
    ROAD_SAFETY_RAW_DATA_PATH: str = WORKING_DIR + "\\assignment1\\data\\road_safety_dataset.arff"

    phishing_data: pd.DataFrame = process_phishing_data(PHISHING_RAW_DATA_PATH)
    road_safety_data: pd.DataFrame = process_road_safety_data(ROAD_SAFETY_RAW_DATA_PATH)

    safety_dataset_condition = CheckDatasetCondition(road_safety_data, ['Casualty_Type'])
    safety_feature_results, safety_target_results = safety_dataset_condition.get_dataset_condition()

    phishing_dataset_condition = CheckDatasetCondition(phishing_data, ['label'])
    phishing_feature_results, phishing_target_results = phishing_dataset_condition.get_dataset_condition()

    print("123")
