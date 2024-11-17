import os

from assignment1.process_data import *

if __name__ == '__main__':
    WORKING_DIR: str = os.getcwd()
    PHISHING_RAW_DATA_PATH: str = WORKING_DIR + "/assignment0/data/PhiUSIIL_Phishing_URL_Dataset.csv"
    ROAD_SAFETY_RAW_DATA_PATH: str = WORKING_DIR + "/assignment1/data/road_safety.csv"

    RUN_CONFIG = {
                'impute_missing': True,
                'remove_vars_multicorr': False,
                'remove_vars_pca': False,
                'dtype_opt': True,
                'road_safety_target': 'Casualty_Severity',
                'phishing_target': 'label'
    }

    # phishing_data: pd.DataFrame = process_phishing_data(PHISHING_RAW_DATA_PATH, RUN_CONFIG)
    # phishing_model_results = fit_svm_model(phishing_data, 'label')

    road_safety_data: pd.DataFrame = process_road_safety_data(ROAD_SAFETY_RAW_DATA_PATH, RUN_CONFIG)
    road_safety_model_results = fit_svm_model(road_safety_data, RUN_CONFIG['road_safety_target'])

    # print('phishing', RUN_CONFIG, phishing_model_results['accuracy'])
    print('road', RUN_CONFIG, road_safety_model_results['accuracy'])

    print("123")
