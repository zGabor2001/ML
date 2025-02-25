from dataclasses import dataclass
from typing import Type

from assignment3.model.base_model import BaseRegressor


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a model to be trained

    Attributes
    ----------
    name : str
        A human-readable name for the model configuration.
    model_cls : Type[BaseRegressor]
        The class of the model to be instantiated and trained.
    parameters : dict[str, list[any]]
        A dictionary with the names of the hyperparameters as keys and a list of possible values as values.
    training_device : str, optional
        The device to be used for training the model.
    """
    name: str
    model_cls: Type[BaseRegressor]
    parameters: dict[str, list[any]]
    fixed_parameters: dict
    training_device: str | None = None
