from os import PathLike
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from assignment2.employee_salaries import prepare_employee_salaries_dataset
from assignment2.model import run_random_forest_with_varied_params, ScratchRandomForest, RFHyperparameters
from assignment2.model.llm_random_forest import LLMRandomForestRegressor
from assignment2.model.runner import run_sklearn_model_with_varied_params
from assignment2.toronto_rental import prepare_toronto_rental_dataset
from assignment2.visualization import plot_parallel_coordinates
from assignment2.visualization.dataframe_to_image import plot_dataframe

_OUTPUT = '../output'
_DATASET_EMPLOYEE_SALARIES = 'employee_salaries'
_DATASET_TORONTO_RENTAL = 'toronto_rental'

_FOLDER_KNN = 'knn'
_FOLDER_RF = 'parameter_permutation'
_FOLDER_PLOTS = 'plots'

_FILE_RF_SCRATCH = 'ScratchRandomForest_results'
_FILE_RF_LLM = 'LLMRandomForestRegressor_results'
_FILE_RF_SKLEARN = 'RandomForestRegressor_results'
_FILE_KNN = 'parameter_permutations.csv'

_FILE_RF_SCRATCH_TOP10 = 'ScratchRandomForest_top10results.csv'
_FILE_RF_LLM_TOP10 = 'LLMRandomForestRegressor_top10results.csv'
_FILE_RF_SKLEARN_TOP10 = 'RandomForestRegressor_top10results.csv'
_FILE_KNN_TOP10 = 'KNeighborsRegressor_top10results.csv'


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

    plot_parallel_coordinates(modified_df, 'RMSE', dataset_name, regressor_name,
                              output_file=filename.with_suffix('.png'))
    plot_parallel_coordinates(modified_df, 'RMSE', dataset_name, regressor_name,
                              output_file=filename_zoomed.with_suffix('.png'), filter_target_range_fraction=0.15)


def read_csv(path: str | PathLike) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    return pd.read_csv(path, index_col=0)


def rf_hyperparameters_from_df_row(row: pd.Series) -> RFHyperparameters:
    return RFHyperparameters(
        no_of_trees=int(row['trees']),
        max_depth=int(row['max_depth']),
        min_samples=int(row['min_samples']),
        feature_subset_size=int(row['feature_subset_size'])
    )


def rerun_models_with_best_hyperparameters(
        dataset_name: str,
        best_rf_scratch: pd.Series,
        best_rf_llm: pd.Series,
        best_rf_sklearn: pd.Series,
        best_knn: pd.Series
):
    if dataset_name is _DATASET_EMPLOYEE_SALARIES:
        (
            x_train_transformed_rf,
            x_test_transformed_rf,
            x_train_transformed_knn,
            x_test_transformed_knn,
            y_train,
            y_test
        ) = prepare_employee_salaries_dataset()
    elif dataset_name is _DATASET_TORONTO_RENTAL:
        (
            x_train_transformed_rf,
            x_test_transformed_rf,
            x_train_transformed_knn,
            x_test_transformed_knn,
            y_train,
            y_test
        ) = prepare_toronto_rental_dataset()
    else:
        raise ValueError('Invalid dataset name')

    folder_rf = Path(_OUTPUT) / dataset_name / _FOLDER_RF
    folder_knn = Path(_OUTPUT) / dataset_name / _FOLDER_KNN
    results_rf_scratch = run_random_forest_with_varied_params(
        model_cls=ScratchRandomForest,
        x_train=x_train_transformed_rf,
        x_test=x_test_transformed_rf,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=[rf_hyperparameters_from_df_row(best_rf_scratch)],
        n_jobs=1,
        n_runs=10,
        verbose=True
    )

    results_rf_scratch.to_csv(folder_rf / _FILE_RF_SCRATCH_TOP10)

    results_rf_llm = run_random_forest_with_varied_params(
        model_cls=LLMRandomForestRegressor,
        x_train=x_train_transformed_rf,
        x_test=x_test_transformed_rf,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=[rf_hyperparameters_from_df_row(best_rf_llm)],
        n_jobs=-1,
        n_runs=10,
        verbose=True
    )

    results_rf_llm.to_csv(folder_rf / _FILE_RF_LLM_TOP10)

    results_rf_sklearn = run_random_forest_with_varied_params(
        model_cls=RandomForestRegressor,
        x_train=x_train_transformed_rf,
        x_test=x_test_transformed_rf,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=[{
            'n_estimators': int(best_rf_sklearn['trees']),
            'max_depth': int(best_rf_sklearn['max_depth']),
            'min_samples_split': int(best_rf_sklearn['min_samples']),
            'max_features': int(best_rf_sklearn['feature_subset_size'])
        }],
        n_jobs=1,
        n_runs=10,
        verbose=True
    )

    results_rf_sklearn.to_csv(folder_rf / _FILE_RF_SKLEARN_TOP10)

    results_knn = run_sklearn_model_with_varied_params(
        model_cls=KNeighborsRegressor,
        x_train=x_train_transformed_knn,
        x_test=x_test_transformed_knn,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=[{
            'n_neighbors': int(best_knn['n_neighbors']),
            'weights': best_knn['weights'],
            'leaf_size': int(best_knn['leaf_size'])
        }],
        n_runs=10,
        verbose=True
    )

    results_knn.to_csv(folder_knn / _FILE_KNN_TOP10)

    return results_rf_scratch, results_rf_llm, results_rf_sklearn, results_knn


def main():
    # create parallel coordinates plots
    dataset_folders = [
        (Path(_OUTPUT) / _DATASET_EMPLOYEE_SALARIES, _DATASET_EMPLOYEE_SALARIES),
        (Path(_OUTPUT) / _DATASET_TORONTO_RENTAL, _DATASET_TORONTO_RENTAL)
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

        # save dataframes to images
        plot_dataframe(df_random_forest_scratch, output_file=folder_rf_plots / 'ScratchRandomForest_results.png',
                       sort_by='RMSE', ascending=True, top_n=10)
        plot_dataframe(df_random_forest_llm, output_file=folder_rf_plots / 'LLMRandomForestRegressor_results.png',
                          sort_by='RMSE', ascending=True, top_n=10)
        plot_dataframe(df_random_forest_sklearn, output_file=folder_rf_plots / 'RandomForestRegressor_results.png',
                            sort_by='RMSE', ascending=True, top_n=10)
        plot_dataframe(df_knn, output_file=folder_knn_plots / 'KNeighborsRegressor_results.png',
                          sort_by='RMSE', ascending=True, top_n=10)

        # create parallel coordinate plots
        create_parallel_coordinates_plot(df_random_forest_scratch, dataset_name, 'ScratchRandomForest', folder_rf_plots)
        create_parallel_coordinates_plot(df_random_forest_llm, dataset_name, 'LLMRandomForestRegressor',
                                         folder_rf_plots)
        create_parallel_coordinates_plot(df_random_forest_sklearn, dataset_name, 'RandomForestRegressor',
                                         folder_rf_plots)
        create_parallel_coordinates_plot(df_knn, dataset_name, 'KNeighborsRegressor', folder_knn_plots)

        # get the best hyperparameters for each model
        best_rf_scratch = df_random_forest_scratch.sort_values(by='RMSE').iloc[0]
        best_rf_llm = df_random_forest_llm.sort_values(by='RMSE').iloc[0]
        best_rf_sklearn = df_random_forest_sklearn.sort_values(by='RMSE').iloc[0]
        best_knn = df_knn.sort_values(by='RMSE').iloc[0]

        # run each model 10 times with the best hyperparameters
        # and save the results
        results_rf_scratch, results_rf_llm, results_rf_sklearn, results_knn = rerun_models_with_best_hyperparameters(
            dataset_name,
            best_rf_scratch,
            best_rf_llm,
            best_rf_sklearn,
            best_knn
        )


if __name__ == '__main__':
    main()
