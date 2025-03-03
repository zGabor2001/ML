import random

import numpy as np

from .config import ModelConfig


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
        self.perf_metrics = None

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> 'CandidateSolution':
        params = [CandidateSolutionParam(name, values) for name, values in model_config.parameters.items()]
        return cls(model_config, params, x_train, y_train, x_test, y_test)

    @property
    def score(self) -> tuple[float, float, float]:
        if self._score is not None:
            return self._score
        model = self.model.model_cls(self.model.training_device)
        kwargs = {param.name: param.current_value for param in self._params}
        model.train(self._x_train, self._y_train, **kwargs)
        predictions = model.predict(self._x_test)
        std_dev, rmse, r2 = model.evaluate(predictions, self._y_test)
        self._score = (std_dev, rmse, r2)
        return std_dev, rmse, r2

    def neighboring_solution(self, neighbor_range: float) -> 'CandidateSolution':
        new_params = self._params.copy()
        param_index = random.randrange(len(self._params))
        new_params[param_index] = self._params[param_index].neighboring_solution(neighbor_range)
        return CandidateSolution(self.model, new_params, self._x_train, self._y_train, self._x_test,
                                 self._y_test)

    def to_dict(self, compute_score: bool = False) -> dict[str, any]:
        std_dev, rmse_score, r2 = self.score
        score = rmse_score if compute_score else self._score
        return ({"model": self.model.name, "score": score, "std_dev": std_dev, "r2": r2} |
                {param.name: param.current_value for param in self._params})
