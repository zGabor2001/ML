from typing import Tuple

from assignment1.utils import (check_missing_values,
                               get_data_types_for_df_columns,
                               check_feature_scaling,
                               detect_outliers_with_iqr,
                               check_categorical_variables,
                               check_correlation_matrix,
                               check_class_balance)
from assignment1.plots import *


class CheckDatasetCondition:
    def __init__(self, df: pd.DataFrame, target_var_name: list):
        self.df_feature = df[[col for col in df.columns if col not in target_var_name]]
        self.df_target = df[target_var_name]
        self.target_var = target_var_name[0]

    @staticmethod
    def _check_features(df: pd.DataFrame) -> dict:
        feature_analysis: dict = {
            'data_types': get_data_types_for_df_columns(df),
            'missing_values': check_missing_values(df),
            'range_and_scale': check_feature_scaling(df),
            'outliers': detect_outliers_with_iqr(df),
            'categorical_vars': check_categorical_variables(df),
            'multicollinearity': check_correlation_matrix(df),
        }
        return feature_analysis

    def _check_target(self, df: pd.DataFrame) -> dict:
        target_analysis: dict = {
            'data_types': get_data_types_for_df_columns(df),
            'class_balance': check_class_balance(df, self.target_var),
            'uniques': df[self.target_var].nunique()
        }
        return target_analysis

    def get_dataset_condition(self) -> Tuple[dict, dict]:
        feature_results: dict = self._check_features(self.df_feature)
        target_results: dict = self._check_target(self.df_target)
        return feature_results, target_results

    def get_plots_for_dataset(self) -> None:
        get_histograms_for_df_numerics(self.df_feature)
        get_boxplots_for_df_numerics(self.df_feature)
        get_pairplots_for_df_numerics(self.df_feature)


