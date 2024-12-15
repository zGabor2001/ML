from os import PathLike
from pathlib import Path

import pandas as pd

from assignment2.visualization import plot_parallel_coordinates

_OUTPUT = '../output'
_FOLDER_EMPLOYEE_SALARIES = 'employee_salaries'
_FOLDER_TORONTO_RENTALS = 'toronto_rental'

_FOLDER_KNN = 'knn'
_FOLDER_RF = 'parameter_permutation'
_FOLDER_PLOTS = 'plots'

_FILE_RF_SCRATCH = 'ScratchRandomForest_results'
_FILE_RF_LLM = 'LLMRandomForestRegressor_results'
_FILE_RF_SKLEARN = 'RandomForestRegressor_results'
_FILE_KNN = 'parameter_permutations.csv'


def create_parallel_coordinates_plot(df: pd.DataFrame, dataset_name: str, regressor_name: str, working_dir: Path):
    """
    Plots a parallel coordinates plot for the given dataframe.
    """
    filename = working_dir / f'{dataset_name}_{regressor_name}_parallel_coordinates'
    filename_zoomed = working_dir / f'{dataset_name}_{regressor_name}_parallel_coordinates_zoomed'

    # create df copy
    modified_df = df.copy()
    # drop Std deviation
    modified_df.drop(columns='Std. Dev.', inplace=True)
    # reorder R_squared and RMSE
    modified_columns = modified_df.columns.tolist()
    modified_columns[-2], modified_columns[-1] = modified_columns[-1], modified_columns[-2]
    modified_df = modified_df[modified_columns]

    plot_parallel_coordinates(modified_df, 'RMSE', dataset_name, regressor_name, output_file=filename.with_suffix('.png'))
    plot_parallel_coordinates(modified_df, 'RMSE', dataset_name, regressor_name, output_file=filename_zoomed.with_suffix('.png'), filter_target_range_fraction=0.15)


def read_csv(path: str | PathLike) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    return pd.read_csv(path, index_col=0)



def main():
    # create parallel coordinates plots
    dataset_folders = [
        (Path(_OUTPUT) / _FOLDER_EMPLOYEE_SALARIES, 'Employee Salaries'),
        (Path(_OUTPUT)/ _FOLDER_TORONTO_RENTALS, 'Toronto Rental')
    ]

    for folder, dataset_name in dataset_folders:
        folder_rf = folder / _FOLDER_RF
        folder_knn = folder / _FOLDER_KNN
        # load data for each regresser pertaining to the dataset
        df_random_forest_scratch = read_csv(folder_rf / _FILE_RF_SCRATCH)
        df_random_forest_llm = read_csv(folder_rf / _FILE_RF_LLM)
        df_random_forest_sklearn = read_csv(folder_rf / _FILE_RF_SKLEARN)
        df_knn = read_csv(folder_knn / _FILE_KNN)

        folder_rf_plots = folder_rf / _FOLDER_PLOTS
        folder_rf_plots.mkdir(parents=True, exist_ok=True)

        folder_knn_plots = folder_knn / _FOLDER_PLOTS
        folder_knn_plots.mkdir(parents=True, exist_ok=True)

        # create parallel coordinate plots
        create_parallel_coordinates_plot(df_random_forest_scratch, dataset_name, 'ScratchRandomForest', folder_rf_plots)
        create_parallel_coordinates_plot(df_random_forest_llm, dataset_name, 'LLMRandomForestRegressor', folder_rf_plots)
        create_parallel_coordinates_plot(df_random_forest_sklearn, dataset_name, 'RandomForestRegressor', folder_rf_plots)
        create_parallel_coordinates_plot(df_knn, dataset_name, 'KNeighborsRegressor', folder_knn_plots)





if __name__ == '__main__':
    main()
