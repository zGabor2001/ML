from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_BASE_FOLDER="results"
_DATASETS={
    "Cars": "cars",
    "Employee salaries": "employee_salaries",
    "Energy efficiency": "energy_efficiency",
    "Toronto Rental": "toronto_rental"
}
_SIMULATED_ANNEALING_FILE="simulated_annealing_results.csv"

def plot_rmse(df: pd.DataFrame, folder: Path):
    rmse_plot = df[['score']]
    rmse_plot = rmse_plot.rename(columns={'score': 'RMSE'})
    std_dev = df["std_dev"][0]
    ax = rmse_plot.plot(marker=None, figsize=(10, 5))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    axhline = ax.axhline(y=std_dev, color='r', linestyle='--', label='Standard Deviation')
    plt.grid()
    file = folder / "rmse_plot.png"
    plt.savefig(file)

    axhline.remove()
    ax.set_yscale('log')
    ax.set_ylabel("Log RMSE")
    file = folder / "rmse_plot_log.png"
    plt.savefig(file)

def plot_r2(df: pd.DataFrame, folder: Path):
    # Plot R2
    r2_plot = df[['r2']]
    r2_plot = r2_plot.rename(columns={'r2': 'R2'})
    ax = r2_plot.plot(marker=None, figsize=(10, 5))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R2")
    plt.grid()
    file = folder / "r2_plot.png"
    plt.savefig(file)
    # Plot Log R2
    ax.set_yscale('log')
    ax.set_ylabel("Log R2")
    file = folder / "r2_plot_log.png"
    plt.savefig(file)


def evaluate():
    script_dir = Path(__file__).parent

    for dataset_name, dataset_folder in _DATASETS.items():
        dataset_folder_path = script_dir / Path(_BASE_FOLDER) / dataset_folder
        simulated_annealing_results = pd.read_csv(str(dataset_folder_path / _SIMULATED_ANNEALING_FILE))
        best_result = simulated_annealing_results.loc[simulated_annealing_results["score"].idxmin()]
        print(f"Best result for {dataset_name} dataset: {best_result}")
        plot_rmse(simulated_annealing_results, dataset_folder_path)
        plot_r2(simulated_annealing_results, dataset_folder_path)


if __name__ == "__main__":
    evaluate()