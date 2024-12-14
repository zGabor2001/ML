import numpy as np
from functools import wraps
import time
from joblib import Parallel, delayed

from assignment2.model.base_random_forest import BaseRandomForest


def timer(func):
    """
    A decorator to measure and print the execution time of a function.

    Args:
    - func (function): The function to be wrapped by the timer decorator.

    Returns:
    - wrapper (function): A wrapped function that calculates and prints the time
                           taken to execute the original function.

    This decorator can be used to wrap functions and output their execution time
    in seconds.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{func.__name__} executed in {duration:.4f} seconds")
        return result

    return wrapper


class DecisionTree:
    def __init__(self,
                 array: np.ndarray,
                 max_depth: int,
                 min_samples: int,
                 task_type: str,
                 target_col_index: int):
        self.array = array
        self.max_depth: int = max_depth
        self.min_samples: int = min_samples
        self.task_type: str = task_type
        self.target_col_index: int = target_col_index
        self.split_metric = None
        self.x = np.delete(self.array, self.target_col_index, axis=1)  # Remove the target column along axis=1
        self.y = self.array[:, self.target_col_index]
        self.tree = None

    def _get_split_metric(self):
        if self.task_type == 'cls':
            raise NotImplementedError("Incorrect split metric type, classification trees "
                                      "are not implemented for this task!")
        return 'var'

    def _is_stop(self, data: np.ndarray, depth: int) -> bool:
        no_of_samples = len(data)
        if (depth >= self.max_depth or
                no_of_samples < self.min_samples or
                len(np.unique(data)) == 1):
            return True
        return False

    def _get_new_leaf(self, y: np.ndarray):
        if self.split_metric == 'cls':
            raise NotImplementedError("Incorrect task type, classification trees are not implemented for this task!")
        return np.mean(y)

    @staticmethod
    def _calculate_variance_reduction(feature: np.ndarray, target: np.ndarray, threshold: float):
        left_indices = feature <= threshold
        right_indices = feature > threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        original_variance = np.var(target)
        left_variance = np.var(target[left_indices]) if np.sum(left_indices) > 0 else 0
        right_variance = np.var(target[right_indices]) if np.sum(right_indices) > 0 else 0

        total_count = target.shape[0]
        left_weight = np.sum(left_indices) / total_count
        right_weight = np.sum(right_indices) / total_count

        weighted_variance = (left_weight * left_variance) + (right_weight * right_variance)
        variance_reduction = original_variance - weighted_variance

        return variance_reduction

    def _best_split(self, features: np.ndarray, target: np.ndarray):
        best_threshold = 0
        best_feature_index = 0
        best_variance_reduction = 0
        for i in range(len(features[0])):
            feature = features[:, i]
            thresholds = np.unique(feature)
            for threshold in thresholds:
                variance_reduction = self._calculate_variance_reduction(feature, target, threshold)
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature_index = i
                    best_threshold = threshold
        return best_feature_index, best_threshold, best_variance_reduction

    def _build_dec_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0):
        if self._is_stop(y, depth):
            return self._get_new_leaf(y)

        best_index, best_threshold, var_reduction = self._best_split(x, y)

        if var_reduction == 0:
            return self._get_new_leaf(y)

        left_indices = x[:, best_index] <= best_threshold  ### maybe self.x
        right_indices = x[:, best_index] > best_threshold

        left_tree = self._build_dec_tree(x[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_dec_tree(x[right_indices], y[right_indices], depth + 1)

        return [left_tree, right_tree, best_index, best_threshold]

    def fit(self):
        self.split_metric = self._get_split_metric()
        self.tree = self._build_dec_tree(self.x, self.y)
        self._is_stop(self.array, 2)

    @staticmethod
    def _traverse_tree(node, sample):
        if not isinstance(node, list):
            return node

        left_tree, right_tree, feature_index, threshold = node

        if sample[feature_index] <= threshold:
            return DecisionTree._traverse_tree(left_tree, sample)
        else:
            return DecisionTree._traverse_tree(right_tree, sample)

    def predict(self, sample: np.ndarray):
        return self._traverse_tree(self.tree, sample)


class ScratchRandomForest(BaseRandomForest):
    def __init__(self,
                 data,
                 no_of_trees: int,
                 max_depth: int,
                 min_samples: int,
                 feature_subset_size: int,
                 task_type: 'str'):
        super().__init__(data, no_of_trees, max_depth, min_samples, feature_subset_size, task_type)
        self.list_of_forests: list = []
        self.unselected_samples: np.ndarray = np.array([])


    @timer
    def _bootstrap_sample(self):
        random_sample_indexes = np.random.randint(0, len(self.data), size=len(self.data))
        sampled: np.ndarray = self.data[random_sample_indexes, :]
        not_sampled: np.ndarray = self.data[np.setdiff1d(np.arange(len(self.data)), random_sample_indexes), :]
        return sampled, not_sampled
    '''
    def _calculate_tree_error(self):
        errors = []
        for tree_index, tree in enumerate(self.list_of_forests):
            sampled, not_sampled = self._bootstrap_sample()

            oob_predictions = tree.predict_batch(not_sampled[:, :-1])
            oob_labels = not_sampled[:, -1]

            if self.task_type == 'reg':
                mse = np.mean((oob_predictions - oob_labels) ** 2)
                errors.append(np.sqrt(mse))
            elif self.task_type == 'cls':
                accuracy = np.mean(oob_predictions == oob_labels)
                errors.append(1 - accuracy)

        return errors
    '''
    @timer
    def fit(self):

        feature_indices_per_tree = [
            np.random.choice(range(self.data.shape[1] - 1), self.feature_subset_size, replace=False)
            for _ in range(self.no_of_trees)
        ]

        # Parallelizable function for training a single tree
        def fit_tree(data: np.ndarray, feature_indices: list[int]):
            random_sample_indices = np.random.randint(0, len(data), size=len(data))
            sampled_rows = data[random_sample_indices, :]
            sampled_features = np.column_stack((sampled_rows[:, feature_indices], sampled_rows[:, -1]))
            dtree = DecisionTree(array=sampled_features,
                                 max_depth=self.max_depth,
                                 min_samples=self.min_samples,
                                 task_type=self.task_type,
                                 target_col_index=-1)
            dtree.fit()
            return dtree, feature_indices

        self.list_of_forests = Parallel(n_jobs=-1)(delayed(fit_tree)(self.data, fi) for fi in feature_indices_per_tree)

    @timer
    def predict(self, samples: np.ndarray):
        tree_predictions = []
        for tree, feature_indices in self.list_of_forests:
            subset_samples = samples[:, feature_indices]
            tree_preds = [tree.predict(sample) for sample in subset_samples]
            tree_predictions.append(tree_preds)

        if self.task_type == 'reg':
            tree_predictions = np.mean(tree_predictions, axis=0)

        return tree_predictions

    @staticmethod
    def evaluate(predictions: np.ndarray, test_labels: np.ndarray):
        mse = np.mean((predictions - test_labels) ** 2)
        return np.sqrt(mse)


'''
def _get_np_array_from_data(self):
    if isinstance(self.data(), np.ndarray):
        return
    elif isinstance(self.data(), pd.DataFrame):
        self.data = self.data.to_numpy()
    elif isinstance(self.data(), dict):
        self.data = np.array(list(self.data.values()))
    else:
        raise TypeError(f"Incorrect type of data input, expected np.array, got {type(self.data)}")
'''
