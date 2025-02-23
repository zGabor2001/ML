from dataclasses import dataclass
from typing import Type

from assignment3.model.base_model import BaseRegressor

@dataclass
class ModelConfig:
    name: str
    model_cls: Type[BaseRegressor]
    parameters: dict[str, list[any]]
    training_device: str | None = None
