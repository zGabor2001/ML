from os import PathLike

import pandas as pd
import plotly.express as px
from typing import Optional
from pathlib import Path


def plot_parallel_coordinates(
        data: pd.DataFrame | str | Path,
        target_col: str,
        dataset_name: str,
        regressor_name: str,
        working_dir: Optional[str | PathLike] = None,
        output_file: Optional[str | PathLike] = None,
        dimensions: Optional[list[str]] = None,
        filter_target_range_fraction: float = 1.0
) -> None:
    """
    Plots a parallel coordinates plot for the given dataframe or CSV file.

    Parameters:
        data (pd.DataFrame | str | Path): The dataframe or path to the CSV file from which the dataframe is loaded.
        target_col (str): The target column for coloring the plot.
        dataset_name (str): The name of the dataset.
        regressor_name (str): The name of the regressor.
        working_dir (str | PathLike, optional): The working directory to save the plot. Defaults to None. Either the output_file or working_dir must be provided.
        output_file (str | PathLike, optional): The output file to save the plot. Defaults to None. Either the output_file or working_dir must be provided.
        dimensions (list[str], optional): The dimensions to plot. Defaults to the columns of the dataframe.
        filter_target_range_fraction (float, optional): The fraction of the target column to filter. Defaults to 1.0.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

    if dimensions is None:
        dimensions = [col for col in df.columns if not 'unnamed' in col.lower()]

    # handle categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_columns:
        # put categorical columns first
        dimensions = categorical_columns + [col for col in dimensions if col not in categorical_columns]

    # map categorical columns to codes
    category_mappings = {}
    for col in categorical_columns:
        df[col] = df[col].astype('category')
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

    # filter the target column
    if 0.0 < filter_target_range_fraction < 1.0:
        threshold_value = df[target_col].min() + (df[target_col].max() - df[target_col].min()) * filter_target_range_fraction
        df = df[df[target_col] <= threshold_value]


    fig = px.parallel_coordinates(
        df,
        color=target_col,
        color_continuous_scale=px.colors.diverging.Tealrose,
        dimensions=dimensions,
        title=f'{dataset_name} - {regressor_name}'
    )

    # map the categorical codes back to their original names
    for i, dim in enumerate(fig.data[0]['dimensions']):
        if dim['label'] in category_mappings:
            mapping = category_mappings[dim['label']]
            dim['tickvals'] = list(mapping.keys())
            dim['ticktext'] = list(mapping.values())

    # determine output file
    if output_file is not None:
        output_file_path = Path(output_file)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    elif working_dir is not None:
        working_dir_path = Path(working_dir)
        working_dir_path.mkdir(parents=True, exist_ok=True)
        output_file = working_dir_path / f'{dataset_name}_{regressor_name}_parallel_coordinates_plot.png'
    else:
        raise ValueError('Either the output_file or working_dir must be provided.')

    fig.write_image(output_file)


if __name__ == "__main__":
    plot_parallel_coordinates(
        data='../output/toronto_rental/knn/parameter_permutations.csv',
        target_col='RMSE',
        dataset_name='Toronto Rental',
        regressor_name='KNN',
        working_dir='../output/toronto_rental/knn',
        filter_target_range_fraction=1.0
    )
