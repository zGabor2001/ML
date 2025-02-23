import random
from typing import TypeVar, Generic

T = TypeVar('T')

class HyperParam(Generic[T]):

    def __init__(self, name: str, values: list[T], initial_value: T = None):
        self.name = name
        self._allowed_values = values

        if initial_value is None:
            self._current_index = random.randrange(len(values))
        elif initial_value not in values:
            raise ValueError(f"Initial value {initial_value} is not in the list of allowed values {values}")
        else:
            self._current_index = values.index(initial_value)

    @property
    def current_value(self) -> T:
        return self._allowed_values[self._current_index]

    def get_neighbor(self, neighbor_range: float) -> 'HyperParam'[T]:
        neighbor_range = int(neighbor_range * len(self._allowed_values))
        new_index = self._current_index + random.randint(-neighbor_range, neighbor_range)
        new_index = max(0, min(new_index, len(self._allowed_values) - 1))
        return self._with_index(new_index)

    def _with_index(self, index: int) -> 'HyperParam'[T]:
        new_param = HyperParam(self.name, self._allowed_values)
        new_param._current_index = index
        return new_param
