import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_histograms_for_df_numerics(df: pd.DataFrame) -> None:
    numerical_cols = df.select_dtypes(include='number')

    for col in numerical_cols.columns:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=30, kde=True, color='teal')
        plt.title(f'Histogram for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        sns.kdeplot(df[col], fill=True, color='teal')
        plt.title(f'KDE Plot for {col}')
        plt.xlabel(col)

        plt.tight_layout()
        plt.show()


def get_boxplots_for_df_numerics(df: pd.DataFrame) -> None:
    numerical_cols = df.select_dtypes(include='number')
    for col in numerical_cols.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=col, color='skyblue')
        plt.title(f'Boxplot for {col}')
        plt.xlabel(col)
        plt.show()
