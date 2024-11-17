import os

import pandas as pd

from assignment1.process_data import *

from assignment1.utils import save_run_results

if __name__ == '__main__':
    WORKING_DIR: str = os.getcwd()
    RESULTS_FILE = os.path.join(WORKING_DIR, 'run_results.csv')

    RUN_CONFIG = {
        'data_path': WORKING_DIR + "/assignment1/data",
        'phishing_data_path': WORKING_DIR + "/assignment0/data/PhiUSIIL_Phishing_URL_Dataset.csv",
        'road_safety_data_path': WORKING_DIR + "/assignment1/data/road_safety.csv",
        'prep_data': False,
        'train_models': True,
        'impute_missing': True,
        'remove_vars_multicorr': False,
        'remove_vars_pca': False,
        'dtype_opt': True,
        'dataset_report': False,
        'phishing_target': 'label',
        'phishing_sample_size': None,   # if None all data is used
        'phishing_kernel_type': 'linear',
        'phishing_model_c': 1.0,
        'road_safety_target': 'Casualty_Severity',
        'road_safety_sample_size': None,    # if None all data is used
        'road_safety_kernel_type': 'linear',
        'road_safety_model_c': 1.0,
    }

    print('RUN_CONFIG:\n')

    for key, value in RUN_CONFIG.items():
        print(key, ' : ', value)

    if RUN_CONFIG['prep_data']:
        phishing_data: pd.DataFrame = process_phishing_data(RUN_CONFIG)
        road_safety_data: pd.DataFrame = process_road_safety_data(RUN_CONFIG)
    else:
        phishing_data: pd.DataFrame = pd.read_csv(
            os.path.join(RUN_CONFIG['data_path'], 'prep_phishing_data.csv')
        )
        road_safety_data: pd.DataFrame = pd.read_csv(
            os.path.join(RUN_CONFIG['data_path'], 'prep_road_data.csv')
        )

    if RUN_CONFIG['train_models']:
        phishing_model_results: dict = fit_svm_model(df=phishing_data,
                                                     target=RUN_CONFIG['phishing_target'],
                                                     kernel=RUN_CONFIG['phishing_kernel_type'],
                                                     c=RUN_CONFIG['phishing_model_c']
                                                     )

        print('phishing', RUN_CONFIG, phishing_model_results['accuracy'])
        save_run_results(RESULTS_FILE, RUN_CONFIG,
                         phishing_model_results['accuracy'], 'phishing')

        road_safety_model_results: dict = fit_svm_model(df=road_safety_data,
                                                        target=RUN_CONFIG['road_safety_target'],
                                                        kernel=RUN_CONFIG['road_safety_kernel_type'],
                                                        c=RUN_CONFIG['road_safety_model_c']
                                                        )

        print('road', RUN_CONFIG, road_safety_model_results['accuracy'])
        save_run_results(RESULTS_FILE, RUN_CONFIG,
                         road_safety_model_results['accuracy'], 'road_safety')
