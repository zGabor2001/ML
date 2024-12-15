from dataclasses import dataclass


@dataclass
class RFHyperparameters:
    no_of_trees: int
    max_depth: int
    min_samples: int
    feature_subset_size: int
    task_type: str = 'reg'


def generate_hyperparameter_permutations(
        no_of_trees: list[int],
        max_depth: list[int],
        min_samples: list[int],
        feature_subset_size: list[int],
        task_type: str = 'reg'
) -> list[RFHyperparameters]:
    return [RFHyperparameters(trees, depth, samples, subset, task_type)
            for trees in no_of_trees
            for depth in max_depth
            for samples in min_samples
            for subset in feature_subset_size]



def generate_knn_hyperparameter_permutations(
        n_neighbors: list[int],
        weights: list[str],
        leaf_size: list[int],
) -> list[dict[str, any]]:
    return [{'n_neighbors': n, 'weights': w, 'leaf_size': l}
            for n in n_neighbors
            for w in weights
            for l in leaf_size]
