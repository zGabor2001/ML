from abc import ABC, abstractmethod

import numpy as np


class BaseRandomForest(ABC):

    def __init__(
            self,
            data: np.ndarray,
            no_of_trees: int,
            max_depth: int,
            min_samples: int,
            feature_subset_size: int,
            task_type: 'str'
    ) -> None:
        self.data = data
        self.no_of_trees: int = no_of_trees
        self.max_depth: int = max_depth
        self.min_samples: int = min_samples
        self.feature_subset_size: int = feature_subset_size
        self.task_type: str = task_type

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(predictions: np.ndarray, test_labels: np.ndarray) -> float:
        pass