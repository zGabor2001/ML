import numpy as np
from typing import List
from assignment2.model.base_random_forest import BaseRandomForest


class DecisionTreeRegressor:
    def _init_(self, max_depth: int, min_samples: int, feature_subset_size: int):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_subset_size = feature_subset_size
        self.tree = None

    def fit(self, data: np.ndarray):
        self.tree = self._build_tree(data, depth=0)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(sample, self.tree) for sample in samples])

    def _build_tree(self, data: np.ndarray, depth: int):
        if depth >= self.max_depth or len(data) <= self.min_samples:
            return {"value": np.mean(data[:, -1])}

        feature_indices = np.random.choice(
            data.shape[1] - 1, self.feature_subset_size, replace=False
        )

        best_split = self._find_best_split(data, feature_indices)
        if not best_split:
            return {"value": np.mean(data[:, -1])}

        left_data, right_data = self._split_data(data, best_split)
        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": self._build_tree(left_data, depth + 1),
            "right": self._build_tree(right_data, depth + 1),
        }

    def _find_best_split(self, data: np.ndarray, feature_indices: np.ndarray):
        best_split = None
        best_mse = float("inf")

        for feature in feature_indices:
            thresholds = np.unique(data[:, feature])
            for threshold in thresholds:
                left_data, right_data = self._split_data(data, {"feature": feature, "threshold": threshold})
                if len(left_data) == 0 or len(right_data) == 0:
                    continue

                mse = self._calculate_mse(left_data, right_data)
                if mse < best_mse:
                    best_mse = mse
                    best_split = {"feature": feature, "threshold": threshold}

        return best_split

    @staticmethod
    def _split_data(data: np.ndarray, split: dict):
        feature, threshold = split["feature"], split["threshold"]
        left_data = data[data[:, feature] <= threshold]
        right_data = data[data[:, feature] > threshold]
        return left_data, right_data

    @staticmethod
    def _calculate_mse(left_data: np.ndarray, right_data: np.ndarray) -> float:
        left_mse = np.var(left_data[:, -1]) * len(left_data)
        right_mse = np.var(right_data[:, -1]) * len(right_data)
        total_mse = (left_mse + right_mse) / (len(left_data) + len(right_data))
        return total_mse

    @staticmethod
    def _predict_sample(sample: np.ndarray, tree: dict):
        if "value" in tree:
            return tree["value"]
        feature, threshold = tree["feature"], tree["threshold"]
        if sample[feature] <= threshold:
            return DecisionTreeRegressor._predict_sample(sample, tree["left"])
        return DecisionTreeRegressor._predict_sample(sample, tree["right"])


class RandomForestRegressor(BaseRandomForest):

    def _init_(
        self,
        data: np.ndarray,
        no_of_trees: int,
        max_depth: int,
        min_samples: int,
        feature_subset_size: int,
        task_type: str = "regression",
    ) -> None:
        super()._init_(
            data, no_of_trees, max_depth, min_samples, feature_subset_size, task_type
        )
        self.trees: List[DecisionTreeRegressor] = []

    def fit(self):
        for _ in range(self.no_of_trees):
            bootstrapped_data = self._bootstrap_sample(self.data)
            tree = DecisionTreeRegressor(
                self.max_depth, self.min_samples, self.feature_subset_size
            )
            tree.fit(bootstrapped_data)
            self.trees.append(tree)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(samples) for tree in self.trees])
        return np.mean(predictions, axis=0)

    @staticmethod
    def evaluate(predictions: np.ndarray, test_labels: np.ndarray) -> float:
     return "mean-square-evaluate"
