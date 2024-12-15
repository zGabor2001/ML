import pandas as pd
import plotly.express as px
from typing import Optional, Union, List, Dict
from pathlib import Path


def plot_parallel_coordinates(
        data: pd.DataFrame | str | Path,
        target_col: str,
        dataset_name: str,
        regressor_name: str,
        working_dir: Union[str, Path],
        dimensions: Optional[list[str]] = None,
        focus_target_quantile: float = 1.0
) -> None:
    """
    Plots a parallel coordinates plot for the given dataframe or CSV file.

    Parameters:
        data (pd.DataFrame | str | Path): The dataframe or path to the CSV file from which the dataframe is loaded.
        target_col (str): The target column for coloring the plot.
        dataset_name (str): The name of the dataset.
        regressor_name (str): The name of the regressor.
        working_dir (Union[str, Path]): The directory to save the plot.
        dimensions (list[str], optional): The dimensions to plot. Defaults to the columns of the dataframe.
        focus_target_quantile (float, optional): The quantile to set as the upper range for target column focus. Defaults to 1.0.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

    if dimensions is None:
        dimensions = {col for col in df.columns if not 'unnamed' in col.lower()}

    # handle categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    category_mappings = {}
    for col in categorical_columns:
        df[col] = df[col].astype('category')
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

    # set the range for the target column
    upper_target_range = df[target_col].quantile(focus_target_quantile)
    lower_target_range = df[target_col].min()

    fig = px.parallel_coordinates(
        df,
        color=target_col,
        color_continuous_scale=px.colors.diverging.Tealrose,
        range_color=[lower_target_range, upper_target_range],
        dimensions=dimensions,
        title=f'{dataset_name} - {regressor_name}'
    )

    # map the categorical codes back to their original names
    for i, dim in enumerate(fig.data[0]['dimensions']):
        if dim['label'] in category_mappings:
            mapping = category_mappings[dim['label']]
            dim['tickvals'] = list(mapping.keys())
            dim['ticktext'] = list(mapping.values())

    working_dir_path = Path(working_dir)
    working_dir_path.mkdir(parents=True, exist_ok=True)
    fig.write_image(working_dir_path / 'parallel_coordinates_plot.png')


if __name__ == "__main__":
    plot_parallel_coordinates(
        data='../output/toronto_rental/knn/parameter_permutations.csv',
        target_col='RMSE',
        dataset_name='Toronto Rental',
        regressor_name='KNN',
        working_dir='../output/toronto_rental/knn',
        focus_target_quantile=0.5
    )
