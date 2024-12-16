from os import PathLike

import pandas as pd
import dataframe_image as dfi


def plot_dataframe(
        df: pd.DataFrame,
        sort_by: str,
        ascending: bool,
        top_n: int,
        output_file: str | PathLike
):
    """
    Plots a dataframe as an image.

    Parameters:
        df (pd.DataFrame): The dataframe to plot.
        sort_by (str): The column to sort the dataframe by.
        ascending (bool): Whether to sort the dataframe in ascending order.
        top_n (int): The number of rows to display.
        output_file (str): The path to save the image.
    """
    modified_df = df.copy()
    modified_df = modified_df.sort_values(sort_by, ascending=ascending)
    modified_df = modified_df.head(top_n)
    modified_df = modified_df.style.format(precision=2).hide(axis='index')
    dfi.export(modified_df, output_file, table_conversion="matplotlib")