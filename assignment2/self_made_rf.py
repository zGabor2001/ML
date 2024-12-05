import numpy as np
import pandas as pd
from functools import wraps
import time


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

    def _get_split_metric(self):
        if self.task_type == 'cls':
            raise NotImplementedError("Incorrect split metric type, classification trees "
                                      "are not implemented for this task!")
        return 'var'

    def _is_stop(self, data: np.ndarray, depth: int) -> bool:
        no_of_samples = len(data)
        if (depth >= self.max_depth or
            no_of_samples < self.min_samples or
            len(np.unique(data)) == 1 or
                no_of_samples < self.min_samples):
            return True
        return False

    def _build_dec_tree(self):
        pass

    def _get_best_split(self):
        pass

    def _get_new_leaf(self, y):
        if self.split_metric == 'cls':
            raise NotImplementedError("Incorrect task type, classification trees are not implemented for this task!")
        return np.mean(y)

    @staticmethod
    def _calculate_variance_reduction(x: np.ndarray):
        return 1, 1

    def _best_split(self, x: np.ndarray):
        best_threshold = 0
        best_feature_index = 0
        for feature in x:
            best_feature_index, threshold = self._calculate_variance_reduction(feature)
            if threshold > best_threshold:
                return best_feature_index, best_threshold
        return best_feature_index, best_threshold

    def _split(self):
        pass

    def fit(self):
        self.split_metric = self._get_split_metric()
        self._is_stop(self.array, 2)

    def predict(self):
        pass


class RandomForest:
    def __init__(self,
                 data,
                 no_of_trees: int,
                 max_depth: int,
                 min_samples: int,
                 feature_subset_size: int,
                 task_type: 'str'):
        self.data = data
        self.no_of_trees: int = no_of_trees
        self.max_depth: int = max_depth
        self.min_samples: int = min_samples
        self.feature_subset_size: int = feature_subset_size
        self.task_type: str = task_type
        self.list_of_forests: list = []
        self.unselected_samples: np.ndarray = np.ndarray([])

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

    @timer
    def _bootstrap_sample(self):
        sampled = np.array([])
        random_sample_indexes = np.random.randint(0, len(self.data), size=self.min_samples)
        sampled: np.ndarray = np.append(sampled, self.data[random_sample_indexes])
        not_sampled: np.ndarray = np.setdiff1d(self.data, sampled)
        return sampled, not_sampled

    @timer
    def _bootstrap_sample_mask(self):
        data_array_length: int = len(self.data)
        random_indices = np.random.randint(0, data_array_length, size=self.min_samples)
        sampled = self.data[random_indices]
        mask = np.ones(data_array_length, dtype=bool)
        mask[random_indices] = False
        not_sampled = self.data[mask]
        return sampled, not_sampled

    def _calculate_tree_error(self):
        pass

    def fit(self):
        bs_sampled, bs_not_sampled = self._bootstrap_sample()
        dtree = DecisionTree(array=bs_sampled,
                             max_depth=self.max_depth,
                             min_samples=self.min_samples,
                             task_type=self.task_type,
                             target_col_index=-1
                             )
        dtree.fit()
        dtree.predict()

    def predict(self):
        pass

    def evaluate(self):
        pass


def run_self_made_random_forest(no_of_trees: int,
                                max_depth: int,
                                min_samples: int,
                                feature_subset_size: int,
                                task_type: 'str'):
    df = pd.DataFrame({'col': list(range(1, 16))})
    data = df.to_numpy()
    rf = RandomForest(data=data,
                      no_of_trees=no_of_trees,
                      max_depth=max_depth,
                      min_samples=min_samples,
                      feature_subset_size=feature_subset_size,
                      task_type=task_type)

    rf.fit()
    print(rf.list_of_forests)
    rf.predict()


if __name__ == '__main__':
    run_self_made_random_forest(no_of_trees=1,
                                max_depth=5,
                                min_samples=10,
                                feature_subset_size=5,
                                task_type='reg')
