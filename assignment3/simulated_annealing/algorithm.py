import math
import random
import time
from datetime import timedelta

import pandas as pd
import numpy as np

from .config import ModelConfig
from .solution import CandidateSolution


class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for automatic model selection and hyperparameter tuning.

    Parameters
    ----------
    model_configs : list[ModelConfig]
        A list of model configurations to be tested by the algorithm.
        A model configuration is a dataclass with the following attributes:
        - name: str
            A human-readable name for the model configuration.
        - model_cls: Type[BaseRegressor]
            The class of the model to be instantiated and trained.
        - parameters: dict[str, list[any]]
            A dictionary with the names of the hyperparameters as keys and a list of possible values as values.
        - training_device: str, optional
            The device to be used for training the model.

    x_train : np.ndarray
        The input features of the training dataset.

    y_train : np.ndarray
        The target values of the training dataset.

    x_test : np.ndarray
        The input features of the test dataset.

    y_test : np.ndarray
        The target values of the test dataset.

    min_temp : float, optional
        The minimum temperature for the annealing process. Set to 0.001 * initial_temperature by default.

    max_time : timedelta, optional
        The maximum time allowed for the algorithm to run. If not specified, the algorithm will run until min_temp or max_steps is reached.

    max_steps : int, optional
        The maximum number of steps (iterations) allowed for the algorithm to run. If not specified, the algorithm will run until min_temp or max_time is reached.

    p_test_different_model : float, optional
        The probability of testing a solution from a different model configuration on each iteration. Default is 0.25.

    neighbor_range : float, optional
        The range of the neighborhood search for each hyperparameter. Default is 0.25.

    initial_temperature : float, optional
        The initial temperature for the annealing process. If not specified, it will be calculated based on the initial acceptance rate.

    initial_acceptance_rate : float, optional
        The initial acceptance rate for the annealing process. Default is 0.8.
        This parameter is used to calculate the initial temperature if it is not specified.

    cooling_factor : float, optional
        The cooling factor for the annealing process. Default is 0.95.

    iterations_per_step : int, optional
        The number of iterations to perform at each step of the annealing process. Default is 10.

    Attributes
    ----------
    best_solution : CandidateSolution
        The best solution found by the algorithm.

    solutions_history_dict : list[dict]
        A list of dictionaries representing the history of solutions found by the algorithm.

    solutions_history_df : pd.DataFrame
        A DataFrame representing the history of solutions found by the algorithm.

    Methods
    -------
    run()
        Run the simulated annealing algorithm and return the best solution found.
    """

    def __init__(
            self,
            model_configs: list[ModelConfig],
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            min_temp: float | None = None,
            max_time: timedelta | None = None,
            max_steps: int | None = None,
            p_test_different_model: float = 0.25,
            neighbor_range: float = 0.25,
            initial_temperature: float | None = None,
            initial_acceptance_rate: float = 0.8,
            cooling_factor: float = 0.95,
            iterations_per_step: int = 10,
    ):
        self.model_configs = model_configs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.min_temp = min_temp
        self.max_time = max_time.total_seconds() if max_time is not None else None
        self.max_steps = max_steps
        self.p_test_different_model = p_test_different_model
        self.neighbor_range = neighbor_range
        self.initial_temperature = initial_temperature
        self.initial_acceptance_rate = initial_acceptance_rate
        self.cooling_factor = cooling_factor
        self.iterations_per_step = iterations_per_step
        self.best_solution: CandidateSolution | None = None
        self._start_time: float = 0.0
        self._step: int = 0
        self._model_solutions: dict[str, CandidateSolution] = {}
        self._current_model: str | None = None
        self._previous_solutions: list[dict[str, any]] = []
        self._temperature: float = 0

    @property
    def current_solution(self) -> CandidateSolution:
        return self._model_solutions[self._current_model]

    @property
    def solutions_history_dict(self) -> list[dict]:
        return self._previous_solutions + [self.current_solution.to_dict(compute_score=True)]

    @property
    def solutions_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.solutions_history_dict)

    def run(self) -> CandidateSolution:
        print("Starting simulated annealing ...")
        self._start_time = time.time()
        self._step = 1
        self._model_solutions = {
            config.name: CandidateSolution.from_model_config(config, self.x_train, self.y_train, self.x_test, self.y_test)
            for config in self.model_configs
        }
        self._current_model = random.choice(list(self._model_solutions.keys()))
        self.best_solution = self.current_solution
        self._previous_solutions = []
        print(f"Initial solution: {self.current_solution.to_dict(compute_score=True)}")

        self._temperature = self.initial_temperature or self._generate_initial_temperature()
        if self.min_temp is None:
            self.min_temp = 0.001 * self._temperature

        while not self._halting_condition():
            print(f"Step: {self._step}, Temperature: {self._temperature:.2f}")
            for i in range(self.iterations_per_step):
                print(f"Iteration {i} at step {self._step}")
                neighbor = self._get_neighboring_solution()
                print(f"Generated new solution: {neighbor.to_dict()}")
                delta = self._calculate_solutions_delta(neighbor)
                if self._accept_solutions_delta(delta):
                    self._previous_solutions.append(self.current_solution.to_dict(compute_score=True))
                    self._model_solutions[neighbor.model.name] = neighbor
                    self._current_model = neighbor.model.name
                    print(f"Accepted new solution: {neighbor.to_dict()}")
                    if neighbor.score < self.best_solution.score:
                        self.best_solution = neighbor
                        print(f"New best solution: {neighbor.to_dict()}")
            self._temperature = self._temperature * self.cooling_factor
            self._step += 1
        print(f"Halting condition reached on final solution: {self.current_solution.to_dict()}")
        return self.best_solution

    def _generate_initial_temperature(self, n_samples: int = 20) -> float:
        print("Generating initial temperature")
        if self.initial_acceptance_rate <= 0 or self.initial_acceptance_rate >= 1:
            raise ValueError("Acceptance rate must be between 0 and 1")
        deltas_sum = 0
        for _ in range(n_samples):
            neighbor = self._get_neighboring_solution()
            delta = self._calculate_solutions_delta(neighbor)
            deltas_sum += math.fabs(delta)
        avg_delta = deltas_sum / n_samples if n_samples > 0 else 0
        temp = -avg_delta / math.log(self.initial_acceptance_rate)
        print(f"Generated initial temperature {temp:.2f}")
        return temp

    def _halting_condition(self) -> bool:
        if self._temperature < self.min_temp:
            print("Minimum temperature reached. Halting now ...")
            return True
        if self.max_steps is not None and self._step > self.max_steps:
            print("Maximum iterations reached. Halting now ...")
            return True
        if self.max_time is not None and time.time() - self._start_time >= self.max_time:
            print("Maximum time reached. Halting now ...")
            return True
        return False

    def _get_neighboring_solution(self) -> CandidateSolution:
        if len(self._model_solutions) > 1 and random.random() < self.p_test_different_model:
            current = random.choice([
                solution for name, solution in self._model_solutions.items() if name != self._current_model
            ])
        else:
            current = self.current_solution
        return current.neighboring_solution(self.neighbor_range)

    def _calculate_solutions_delta(self, neighbor_solution: CandidateSolution) -> float:
        current_rmse = self.current_solution.score
        neighbor_rmse = neighbor_solution.score
        return current_rmse - neighbor_rmse

    def _accept_solutions_delta(self, score_delta: float) -> bool:
        return score_delta > 0 or (
                self._temperature > 0 and random.random() < math.exp(score_delta / self._temperature))
