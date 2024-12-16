from os import PathLike

import pandas as pd
import plotly.express as px


def plot_boxplot(
        df: pd.DataFrame,
        category: str,
        value: str,
        title: str,
        output_file: str | PathLike
):
    """
    Plots a boxplot for the given dataframe.
    
    Parameters:
        df (pd.DataFrame): The dataframe to plot.
        category (str): The column to group by.
        value (float): The column to plot.
        title (str): The title of the plot.
        output_file (str | PathLike): The path to save the image.
    """
    fig = px.box(df, x=category, y=value, title=title, points='suspectedoutliers')
    fig.write_image(output_file)
    

if __name__  == '__main__':
    df_knn = pd.read_csv('../../output/employee_salaries/knn/KNeighborsRegressor_top10results.csv', index_col=0)
    df_knn['regressor'] = 'KNN'
    df_rf_scratch = pd.read_csv('../../output/employee_salaries/parameter_permutation/ScratchRandomForest_top10results.csv', index_col=0)
    df_rf_scratch['regressor'] = 'RF Scratch'
    df_rf_llm = pd.read_csv('../../output/employee_salaries/parameter_permutation/LLMRandomForestRegressor_top10results.csv', index_col=0)
    df_rf_llm['regressor'] = 'RF LLM'
    df_rf_sklearn = pd.read_csv('../../output/employee_salaries/parameter_permutation/RandomForestRegressor_top10results.csv', index_col=0)
    df_rf_sklearn['regressor'] = 'RF Sklearn'

    df_rf = pd.concat([df_rf_scratch, df_rf_llm, df_rf_sklearn])
    df_all = pd.concat([df_knn, df_rf])
    plot_boxplot(df_rf, 'regressor', 'RMSE', 'Top 10 RMSE results', '../../output/employee_salaries/boxplot_RMSE_RF.png')
    plot_boxplot(df_all, 'regressor', 'RMSE', 'Top 10 RMSE results', '../../output/employee_salaries/boxplot_RMSE_all.png')


    plot_boxplot(df_rf, 'regressor', 'R_squared', 'Top 10 RMSE results', '../../output/employee_salaries/boxplot_RMSE_RF.png')
    plot_boxplot(df_all, 'regressor', 'R_squared', 'Top 10 RMSE results', '../../output/employee_salaries/boxplot_RMSE_all.png')

