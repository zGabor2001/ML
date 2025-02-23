import random
from typing import Type

from assignment3.model.base_model import BaseRegressor
from assignment3.simulated_annealing.parameter import HyperParam


class CandidateModel:

    def __init__(self, name: str, model: Type[BaseRegressor], params: list[HyperParam], device: str = None):
        self.name = name
        self._model_cls = model
        self._params = params
        self._score = None
        self._device = device

    @classmethod
    def from_params_dict(cls, name: str, model: Type[BaseRegressor], params_dict: dict[str, any], device: str = None) -> 'CandidateModel':
        params = [HyperParam(name, values) for name, values in params_dict]
        return cls(name, model, params, device)

    def score(self, x_train, y_train, y_test) -> float:
        if self._score is not None:
            return self._score

        model = self._model_cls(self._device)
        kwargs = self.get_parameter_dict()
        model.train(X_train=x_train, y_train=y_train, **kwargs)
        predictions = model.predict(x_train)
        self._score = model.evaluate(predictions, y_test)
        return self._score

    def get_neighboring_candidate(self, neighbor_range: float, n_params_to_change: int = 1) -> 'CandidateModel':
        if n_params_to_change > len(self._params):
            raise ValueError(f"Cannot change more than {len(self._params)} parameters for model {self.name}")

        params_copy = self._params.copy()
        param_indices = random.sample(range(len(self._params)), n_params_to_change)
        for i in param_indices:
            params_copy[i] = self._params[i].get_neighbor(neighbor_range)
        return CandidateModel(self.name, self._model_cls, params_copy, self._device)

    def get_parameter_dict(self) -> dict[str, any]:
        return {param.name: param.current_value for param in self._params}

    def to_dict(self) -> dict[str, any]:
        return {"model": self.name} | self.get_parameter_dict()



