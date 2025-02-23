import math
import random
import time
import logging
from datetime import timedelta

import pandas as pd
import numpy as np

from assignment3.simulated_annealing import ModelConfig
from assignment3.simulated_annealing.solution import CandidateSolution


class SimulatedAnnealing:
    def __init__(
            self,
            model_configs: list[ModelConfig],
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            min_temp: float | None = None,
            max_time: timedelta | None = None,
            max_iter: int | None = None,
            p_test_different_model: float = 0.25,
            neighbor_range: float = 0.25,
            initial_temperature: float | None = None,
            initial_acceptance_rate: float = 0.8,
            cooling_factor: float = 0.95,
            iterations_per_temp: int = 100,
    ):
        self.model_configs = model_configs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.min_temp = min_temp
        self.max_time = max_time.total_seconds() if max_time is not None else None
        self.max_iter = max_iter
        self.p_test_different_model = p_test_different_model
        self.neighbor_range = neighbor_range
        self.initial_temperature = initial_temperature
        self.initial_acceptance_rate = initial_acceptance_rate
        self.cooling_factor = cooling_factor
        self.iterations_per_temp = iterations_per_temp
        self.best_solution: CandidateSolution | None = None
        self._start_time: float = 0.0
        self._tick: int = 0
        self._model_solutions: dict[ModelConfig, CandidateSolution] = {}
        self._current_model: ModelConfig | None = None
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
        logging.info("Starting simulated annealing ...")
        self._start_time = time.time()
        self._tick = 0
        self._model_solutions = {
            config: CandidateSolution.from_model_config(config, self.x_train, self.y_train, self.x_test, self.y_test)
            for config in self.model_configs
        }
        self._current_model = random.choice(self.model_configs)
        self.best_solution = self.current_solution
        self._previous_solutions = []
        logging.debug(f"Initial solution: {self.current_solution.to_dict(compute_score=True)}")

        self._temperature = self.initial_temperature or self._generate_initial_temperature()
        if self.min_temp is None:
            self.min_temp = 0.001 * self._temperature

        while not self._halting_condition():
            logging.info(f"Tick: {self._tick}, Temperature: {self._temperature:.2f}")
            for i in range(self.iterations_per_temp):
                logging.debug(f"Iteration {i} at tick {self._tick}")
                neighbor = self._get_neighboring_solution()
                logging.debug(f"Generated new solution: {neighbor.to_dict()}")
                delta = self._calculate_solutions_delta(neighbor)
                if self._accept_solutions_delta(delta):
                    self._previous_solutions.append(self.current_solution.to_dict(compute_score=True))
                    self._model_solutions[neighbor.model] = neighbor
                    self._current_model = neighbor.model
                    logging.debug(f"Accepted new solution: {neighbor.to_dict()}")
                    if neighbor.score < self.best_solution.score:
                        self.best_solution = neighbor
                        logging.debug(f"New best solution: {neighbor.to_dict()}")
            self._temperature = self._temperature * self.cooling_factor
            self._tick += 1
        logging.info(f"Halting condition reached on final solution: {self.current_solution.to_dict()}")
        return self.best_solution

    def _generate_initial_temperature(self, n_samples: int = 20) -> float:
        logging.debug("Generating initial temperature")
        if self.initial_acceptance_rate <= 0 or self.initial_acceptance_rate >= 1:
            raise ValueError("Acceptance rate must be between 0 and 1")
        deltas_sum = 0
        for _ in range(n_samples):
            neighbor = self._get_neighboring_solution()
            delta = self._calculate_solutions_delta(neighbor)
            deltas_sum += math.fabs(delta)
        avg_delta = deltas_sum / n_samples if n_samples > 0 else 0
        temp = -avg_delta / math.log(self.initial_acceptance_rate)
        logging.debug(f"Generated initial temperature {temp:.2f}")
        return temp

    def _halting_condition(self) -> bool:
        if self._temperature < self.min_temp:
            logging.debug("Minimum temperature reached. Halting now ...")
            return True
        if self.max_iter is not None and self._tick >= self.max_iter:
            logging.debug("Maximum iterations reached. Halting now ...")
            return True
        if self.max_time is not None and time.time() - self._start_time >= self.max_time:
            logging.debug("Maximum time reached. Halting now ...")
            return True
        return False

    def _get_neighboring_solution(self) -> CandidateSolution:
        if len(self._model_solutions) > 1 and random.random() < self.p_test_different_model:
            current = random.choice([
                solution for model, solution in self._model_solutions.items() if model != self._current_model
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
