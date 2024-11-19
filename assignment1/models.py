from typing import Dict

import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from functools import wraps


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

@timer
def run_models(X, Y, RANDOM_STATE):
    holdout_X_train, holdout_X_test, holdout_Y_train, holdout_Y_test = train_test_split(X, Y, test_size=0.3,
                                                                                        random_state=RANDOM_STATE)
    cross_validation_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def get_metrics_dict(
            accuracy: float,
            f1: float,
            precision: float,
            recall: float,
    ) -> Dict[str, float]:
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    @timer
    def find_best_estimator(
            classifier,
            param_grid: dict,
            cv: int = 5
    ) -> GridSearchCV:
        grid_search = GridSearchCV(
            classifier,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy"
        )
        grid_search.fit(holdout_X_train, holdout_Y_train)
        return grid_search.best_estimator_

    @timer
    def run_random_forest(classifier: RandomForestClassifier | None = None) -> list[dict[str, any]]:
        if classifier is None:
            classifier = RandomForestClassifier()

        classifier.set_params(random_state=RANDOM_STATE)

        # Holdout method
        classifier.fit(holdout_X_train, holdout_Y_train)
        holdout_y_pred = classifier.predict(holdout_X_test)
        holdout_results = get_metrics_dict(
            accuracy=accuracy_score(holdout_Y_test, holdout_y_pred),
            f1=f1_score(holdout_Y_test, holdout_y_pred, average='macro'),
            precision=precision_score(holdout_Y_test, holdout_y_pred, average='macro'),
            recall=recall_score(holdout_Y_test, holdout_y_pred, average='macro'),
        )

        # Cross-validation
        cv_scores = cross_validate(classifier, X, Y, cv=cross_validation_split,
                                   scoring=['accuracy', 'f1', 'precision', 'recall'])
        cv_results = get_metrics_dict(
            accuracy=cv_scores['test_accuracy'].mean(),
            f1=cv_scores['test_f1'].mean(),
            precision=cv_scores['test_precision'].mean(),
            recall=cv_scores['test_recall'].mean(),
        )

        common_results = {
            "classifier": "Random Forest",
            "n_estimators": classifier.n_estimators,
            "max_depth": classifier.max_depth,
            "min_samples_split": classifier.min_samples_split,
            "min_samples_leaf": classifier.min_samples_leaf,
        }

        return [
            {
                **common_results,
                "Data Split": "Holdout",
                **holdout_results
            },
            {
                **common_results,
                "Data Split": "Cross Validation",
                **cv_results
            }
        ]

    rf_classifiers = [
        RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
        RandomForestClassifier(n_estimators=200, min_samples_split=4, min_samples_leaf=1),
        RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_depth=15),
        RandomForestClassifier(n_estimators=150, min_samples_split=5, min_samples_leaf=2, max_depth=20),
        RandomForestClassifier(n_estimators=250, min_samples_split=3, min_samples_leaf=3, max_depth=10)
    ]

    rf_results = []
    for classifier in rf_classifiers:
        rf_results.extend(run_random_forest(classifier))  # Assumes run_random_forest is defined elsewhere

    rf_results_df = pd.DataFrame(rf_results)
    rf_results_df.sort_values(by='accuracy', ascending=False).round(3)

    rf_param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
    }

    best_rf = find_best_estimator(
        classifier=RandomForestClassifier(),
        param_grid=rf_param_grid,
        cv=5
    )

    best_rf_results = pd.DataFrame(run_random_forest(best_rf))

    @timer
    def run_mlp(classifier: MLPClassifier | None = None) -> list[dict[str, any]]:
        if classifier is None:
            classifier = MLPClassifier()

        # create a pipeline which both scales data using standard scaler and then estimates using MLP
        classifier.set_params(random_state=RANDOM_STATE)
        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('mlp', classifier),
        ])
        # holdout method
        pipeline.fit(holdout_X_train, holdout_Y_train)
        holdout_y_pred = pipeline.predict(holdout_X_test)

        holdout_results = get_metrics_dict(
            accuracy=accuracy_score(holdout_Y_test, holdout_y_pred),
            f1=f1_score(holdout_Y_test, holdout_y_pred, average='macro'),
            precision=precision_score(holdout_Y_test, holdout_y_pred, average='macro'),
            recall=recall_score(holdout_Y_test, holdout_y_pred, average='macro'),
        )

        # cross validation
        cv_scores = cross_validate(pipeline, X, Y, cv=cross_validation_split,
                                   scoring=['accuracy', 'f1', 'precision', 'recall'])
        cv_results = get_metrics_dict(
            accuracy=cv_scores['test_accuracy'].mean(),
            f1=cv_scores['test_f1'].mean(),
            precision=cv_scores['test_precision'].mean(),
            recall=cv_scores['test_recall'].mean(),
        )

        common_results = {
            "classifier": "MLP",
            "hidden_layer_sizes": classifier.hidden_layer_sizes,
            "max_iter": classifier.max_iter,
            "activation": classifier.activation,
            "solver": classifier.solver,
        }

        return [
            {
                **common_results,
                "Data Split": "Holdout",
                **holdout_results
            },
            {
                **common_results,
                "Data Split": "Cross Validation",
                **cv_results
            }
        ]

    mlp_classifiers = [
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=200),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200),
        MLPClassifier(hidden_layer_sizes=(200,), max_iter=300, activation="logistic"),
        MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=300, solver="lbfgs"),
        MLPClassifier(hidden_layer_sizes=(300,), max_iter=500, activation="identity")
    ]

    mlp_results = []
    for classifier in mlp_classifiers:
        mlp_results.extend(run_mlp(classifier))

    mlp_results_df = pd.DataFrame(mlp_results)
    mlp_results_df.sort_values(by='accuracy', ascending=False).round(3)

    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (200,), (100, 50), (100, 50, 25)],
        'max_iter': [200, 300, 500],
        'activation': ['relu', 'tanh', 'logistic'],  # Optional for activation exploration
        'solver': ['adam', 'sgd'],  # Optional for solver exploration
    }

    best_mlp = find_best_estimator(
        classifier=MLPClassifier(),
        param_grid=mlp_param_grid,
        cv=5
    )

    best_mlp_results = pd.DataFrame(run_mlp(best_mlp))

    @timer
    def run_svc(classifier: SVC | None = None) -> list[dict[str, any]]:
        if classifier is None:
            classifier = SVC()

        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('svc', classifier),
        ])
        # Holdout method
        pipeline.fit(holdout_X_train, holdout_Y_train)
        holdout_y_pred = pipeline.predict(holdout_X_test)
        holdout_results = get_metrics_dict(
            accuracy=accuracy_score(holdout_Y_test, holdout_y_pred),
            f1=f1_score(holdout_Y_test, holdout_y_pred, average='macro'),
            precision=precision_score(holdout_Y_test, holdout_y_pred, average='macro'),
            recall=recall_score(holdout_Y_test, holdout_y_pred, average='macro'),
        )

        # Cross-validation
        cv_scores = cross_validate(pipeline, X, Y, cv=cross_validation_split,
                                   scoring=['accuracy', 'f1', 'precision', 'recall'])
        cv_results = get_metrics_dict(
            accuracy=cv_scores['test_accuracy'].mean(),
            f1=cv_scores['test_f1'].mean(),
            precision=cv_scores['test_precision'].mean(),
            recall=cv_scores['test_recall'].mean(),
        )

        common_results = {
            "classifier": "SVC",
            "kernel": classifier.kernel,
            "C": classifier.C,
            "gamma": classifier.gamma,
            "degree": classifier.degree,
            "coef0": classifier.coef0
        }

        return [
            {
                **common_results,
                "Data Split": "Holdout",
                **holdout_results
            },
            {
                **common_results,
                "Data Split": "Cross Validation",
                **cv_results
            }
        ]

    svc_classifiers = [
        SVC(kernel='linear', C=0.1, gamma='scale'),
        SVC(kernel='rbf', C=1.0, gamma=0.1),
        SVC(kernel='poly', degree=2, C=1.0, gamma='auto', coef0=0.0),
        SVC(kernel='poly', degree=3, C=10.0, gamma='scale', coef0=1.0),
        SVC(kernel='sigmoid', C=0.5, gamma=0.01, coef0=0.5)
    ]

    svc_results = []
    for classifier in svc_classifiers:
        svc_results.extend(run_svc(classifier))

    svc_results_df = pd.DataFrame(svc_results)
    svc_results_df.sort_values(by='accuracy', ascending=False).round(3)

    svc_param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.1],
        'degree': [2, 3],
        'coef0': [0.0, 0.5]
    }

    best_svc = find_best_estimator(
        classifier=SVC(),
        param_grid=svc_param_grid,
        cv=5
    )

    best_svc_results = pd.DataFrame(run_svc(best_svc))

    results = pd.concat(
        [rf_results_df, mlp_results_df, svc_results_df, best_rf_results, best_mlp_results, best_svc_results],
        join='inner')
    results.sort_values(by='accuracy', ascending=False).round(3)

    return results
