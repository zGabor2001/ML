import random


from assignment3.simulated_annealing import ModelConfig

class CandidateSolutionParam:

    def __init__(self, name: str, values: list[any], current_index: int | None = None):
        self.name = name
        self._values = values
        if current_index is None:
            self._current_index = random.randrange(len(values))
        else:
            if current_index > len(values) - 1:
                raise ValueError(
                    f'Index of current value {current_index} is outside the range of available values 0 to {len(values)}')
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

    # TODO: add type hints for test and train datasets
    def __init__(self, model_config: ModelConfig, params: list[CandidateSolutionParam], x_train, y_train, x_test, y_test):
        self._model_config = model_config
        self._params = params
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._score = None


    @classmethod
    def from_model_config(cls, model_config: ModelConfig, x_train, y_train, x_test, y_test) -> 'CandidateSolution':
        params = [CandidateSolutionParam(name, values) for name, values in model_config.parameters.items()]
        return cls(model_config, params, x_train, y_train, x_test, y_test)

    @property
    def score(self) -> float:
        if self._score is not None:
            return self._score
        model = self._model_config.model_cls(self._model_config.training_device)
        model.train(self._x_train, self._y_train)

        kwargs = {param.name: param.current_value for param in self._params}
        # TODO: add kwargs to BaseRegressor
        model.predict(self._x_test, **kwargs)
        result = model.evaluate(self._x_test, self._y_test)
        self._score = result
        return result


    def neighboring_solution(self, neighbor_range: float) -> 'CandidateSolution':
        new_params = self._params.copy()
        param_index = random.randrange(len(self._params))
        new_params[param_index] = self._params[param_index].neighboring_solution(neighbor_range)
        return CandidateSolution(self._model_config, new_params, self._x_train, self._y_train, self._x_test, self._y_test)
