import math
import random
import time
from datetime import timedelta

from assignment3.simulated_annealing.candidate import CandidateModel


class SimulatedAnnealing:
    def __init__(
            self,
            search_space: list[CandidateModel],
            x_train,
            y_train,
            y_test,
            p_test_different_model: float = 0.25,
            neighbor_range: float = 0.1,
            initial_temperature: float | None = None,
            initial_acceptance_rate: float = 0.8,
            cooling_factor: float = 0.95,
            iterations_per_temp: int = 100,
            halting_min_temp: float | None = None,
            halting_max_iter: int | None = None,
            halting_max_time: timedelta | None = None,
    ):

        if halting_min_temp is None and halting_max_iter is None and halting_max_time is None:
            raise ValueError("At least one of halting_min_temp, halting_max_iter, or halting_max_time must be provided")

        self._solution_candidates = {model.name: model for model in search_space}
        if len(self._solution_candidates) < len(search_space):
            names = [model.name for model in search_space]
            duplicates = {name for name in names if names.count(name) > 1}
            raise ValueError("Duplicate model names found in search space: " + ", ".join(duplicates))
        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test
        self.p_switch_model = p_test_different_model
        self.neighbor_range = neighbor_range
        self.initial_temperature = initial_temperature
        self.initial_acceptance_rate = initial_acceptance_rate
        self.cooling_factor = cooling_factor
        self.iterations_per_temp = iterations_per_temp
        self.halting_min_temp = halting_min_temp
        self.halting_max_iter = halting_max_iter
        self.halting_max_time = halting_max_time.total_seconds() \
            if halting_max_time is not None else None

        self._tick = None
        self._start_time = None
        self._temperature = None
        self._current_solution_index = None
        self._best_solution = None
        self._previous_solutions = []

    @property
    def current_solution(self) -> CandidateModel:
        return self._solution_candidates[self._current_solution_index]

    def run(self) -> CandidateModel:
        print("Starting Simulated Annealing")
        self._start_time = time.time()
        self._current_solution_index = random.choice(list(self._solution_candidates.keys()))
        self._best_solution = self.current_solution
        print(f"Initial solution: {self.current_solution.to_dict()}")
        self._tick = 0
        self._temperature = self.initial_temperature or self._generate_initial_temperature()

        while not self._halting_condition():
            print(f"Tick: {self._tick}, Temperature: {self._temperature:.2f}")
            for i in range(self.iterations_per_temp):
                print(f"Iteration {i} at tick {self._tick}")
                neighbor = self._get_neighboring_solution()
                delta = self._calculate_solutions_delta(neighbor)
                print(f"Generated new solution: {neighbor.to_dict()}")
                if self._accept_solutions_delta(delta):
                    self._previous_solutions.append(self.current_solution)
                    self._solution_candidates[neighbor.name] = neighbor
                    self._current_solution_index = neighbor.name
                    print(f"Accepted new solution: {neighbor.to_dict()}")
                    # TODO: update the best solution
            self._temperature = self._temperature * self.cooling_factor
            self._tick += 1
        print(f"Halting condition reached on final solution: {self.current_solution.to_dict()}")
        return self.current_solution

    def _generate_initial_temperature(self, n_samples: int = 20) -> float:
        print("Generating initial temperature")
        if self.initial_acceptance_rate <= 0 or self.initial_acceptance_rate >= 1:
            raise ValueError("Acceptance rate must be between 0 and 1")
        deltas_sum = 0
        for _ in range(n_samples):
            neighbor = self._get_neighboring_solution()
            delta = self._calculate_solutions_delta(neighbor)
            deltas_sum += delta
        avg_delta = deltas_sum / n_samples
        temp = -avg_delta / math.log(self.initial_acceptance_rate)
        print(f"Generated initial temperature {temp:.2f}")
        return temp

    def _halting_condition(self) -> bool:
        if self.halting_min_temp is not None and self._temperature < self.halting_min_temp:
            return True
        if self.halting_max_iter is not None and self._tick >= self.halting_max_iter:
            return True
        if self.halting_max_time is not None and time.time() - self._start_time >= self.halting_max_time:
            return True
        return False

    def _get_neighboring_solution(self) -> CandidateModel:
        if random.random() < self.p_switch_model:
            model = random.choice([model for name, model in self._solution_candidates if name != self._current_solution_index])
        else:
            model = self.current_solution
        return model.get_neighboring_candidate(self.neighbor_range)

    def _calculate_solutions_delta(self, neighbor_solution: CandidateModel) -> float:
        current_rmse = self.current_solution.score(self.x_train, self.y_train, self.y_test)
        neighbor_rmse = neighbor_solution.score(self.x_train, self.y_train, self.y_test)
        return current_rmse - neighbor_rmse

    def _accept_solutions_delta(self, score_delta: float) -> bool:
        return score_delta > 0 or (
                self._temperature > 0 and random.random() < math.exp(score_delta / self._temperature))
