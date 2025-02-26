import random

import numpy as np

from assignment3.simulated_annealing.config import ModelConfig


class CandidateSolutionParam:

    def __init__(self, name: str, values: list[any], current_index: int | None = None):
        self.name = name
        self._values = values
        if current_index is None:
            self._current_index = random.randrange(len(values))
        else:
            if current_index > len(values) - 1:
                raise ValueError(
                    f'Index of current value {current_index} is outside the range of available values 0 to {len(values) - 1}')
            self._current_index = current_index

    @property
    def current_value(self) -> any:
        return self._values[self._current_index]

    def neighboring_solution(self, neighbor_range: float) -> 'CandidateSolutionParam':
        range_values = int(neighbor_range * len(self._values))
        min_val = max(0, self._current_index - range_values)
        max_val = min(len(self._values) - 1, self._current_index + range_values)
        neighbor_index = random.randrange(min_val, max_val + 1)
        return CandidateSolutionParam(self.name, self._values, neighbor_index)


class CandidateSolution:

    def __init__(self, model_config: ModelConfig, params: list[CandidateSolutionParam], x_train: np.ndarray,
                 y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        self.model = model_config
        self._params = params
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._score = None

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> 'CandidateSolution':
        tunable_params = [
            CandidateSolutionParam(name, values)
            for name, values in model_config.parameters.items()
        ]

        fixed_params = []
        if hasattr(model_config, 'fixed_parameters') and model_config.fixed_parameters:
            for name, value in model_config.fixed_parameters.items():
                fixed_params.append(CandidateSolutionParam(name, [value]))

        all_params = tunable_params + fixed_params

        return cls(model_config, all_params, x_train, y_train, x_test, y_test)

    @property
    def score(self) -> tuple[float, float, float]:
        if self._score is not None:
            return self._score
        fixed_kwargs = getattr(self.model, 'fixed_parameters', {})
        tunable_kwargs = {param.name: param.current_value for param in self._params if param.name not in fixed_kwargs}

        model = self.model.model_cls(**fixed_kwargs, device=self.model.training_device)
        model.train(self._x_train, self._y_train, **tunable_kwargs)
        predictions = model.predict(self._x_test)
        _, rmse, _ = model.evaluate(predictions, self._y_test)
        self._score = rmse
        return rmse

    def neighboring_solution(self, neighbor_range: float) -> 'CandidateSolution':
        new_params = self._params.copy()
        param_index = random.randrange(len(self._params))
        new_params[param_index] = self._params[param_index].neighboring_solution(neighbor_range)
        return CandidateSolution(self.model, new_params, self._x_train, self._y_train, self._x_test,
                                 self._y_test)

    def to_dict(self, compute_score: bool = False) -> dict[str, any]:
        score = self.score if compute_score else self._score
        return ({"model": self.model.name, "score": score} |
                {param.name: param.current_value for param in self._params})
