import pandas as pd
from typing import Tuple

from assignment1.utils import (check_missing_values,
                               get_data_types_for_df_columns)


class CheckDatasetCondition:
    def __init__(self, df: pd.DataFrame):
        self.df_feature = df[:-1]
        self.df_target = df[-1]

    @staticmethod
    def _check_features(df: pd.DataFrame) -> dict:
        #### Feature selection!!!!!!!!!!!!!!!!
        feature_analysis: dict = {
            'data_types': get_data_types_for_df_columns(df),
            'missing_values': check_missing_values(df),
            'range_and_scale': {},
            'outliers': {},
            'categorical_vars': {},
            'multicollinearity': {}
        }
        return feature_analysis

    @staticmethod
    def _check_target(df: pd.DataFrame) -> dict:
        target_analysis: dict = {
            'data_types': {},
            'class_balance': {},
            'uniques': {}
        }
        return target_analysis

    def get_dataset_condition(self) -> Tuple[dict, dict]:
        feature_results: dict = self._check_features(self.df_feature)
        target_results: dict = self._check_target(self.df_target)
        return feature_results, target_results
