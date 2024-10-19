import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_numeric_cols(df) -> list:
    input_variables = []
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            input_variables.append(col)
    return input_variables


def get_histograms_for_numeric_data(working_dir: str, df: pd.DataFrame, num_cols: list, dataset_name: str) -> None:
    total_cols = len(num_cols)
    index = 0

    while index < total_cols:
        num_to_plot = min(4, total_cols - index)
        num_rows = (num_to_plot + 1) // 2

        fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
        axes = axes.flatten()

        for i in range(num_to_plot):
            col = num_cols[index + i]
            sns.histplot(df[col], bins=100, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].set_aspect(aspect='auto')

        plt.tight_layout()
        os.makedirs(os.path.join(working_dir, 'histograms'), exist_ok=True)
        plt.savefig(f'{working_dir}\\histograms\\{dataset_name}_histograms_subplot_{index // 4 + 1}.png')
        plt.close(fig)

        index += num_to_plot
