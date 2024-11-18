import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed

from assignment1.utils import timer


class SupportVectorMachineClassifier:
    def __init__(self, kernel='linear', C=1.0):
        """Initialize the SVM classifier.

        Parameters:
        - kernel: str, optional (default='linear')
            Specifies the kernel type to be used in the algorithm.
        - C: float, optional (default=1.0)
            Regularization parameter.
        """
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = []

    @timer
    def preprocess_data(self, df, target_column):
        """Preprocess the input DataFrame.

        Parameters:
        - df: pd.DataFrame
            The input data.
        - target_column: str
            The name of the target column.

        Returns:
        - X_train, X_test, y_train, y_test: np.ndarray
            Preprocessed training and testing data.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @timer
    def fit_model(self, X_train, y_train):
        model = SVC(kernel=self.kernel, C=self.C)
        model.fit(X_train, y_train)
        return model

    @timer
    def fit(self, X_train, y_train, n_jobs=-1):
        n_splits = 10
        X_splits = np.array_split(X_train, n_splits)
        y_splits = np.array_split(y_train, n_splits)

        self.models = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_model)(X_splits[i], y_splits[i]) for i in range(n_splits)
        )

    @timer
    def predict(self, X):
        """Predict using the trained SVM model.

        Parameters:
        - X: np.ndarray
            Input features for prediction.

        Returns:
        - y_pred: np.ndarray
            Predicted labels.
        """
        predictions = np.mean([model.predict(X) for model in self.models], axis=0)
        return np.round(predictions)

    @timer
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.

        Parameters:
        - X_test: np.ndarray
            Testing features.
        - y_test: np.ndarray
            True labels for the test set.

        Returns:
        - dict: Evaluation results containing accuracy and a classification report.
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    @timer
    def cross_validate(self, df, target_column, cv_splits=10, n_jobs=-1):
        """Perform cross-validation on the SVM model.

        Parameters:
        - df: pd.DataFrame
            The input data.
        - target_column: str
            The name of the target column.
        - cv_splits: int, optional (default=10)
            The number of splits for cross-validation.
        - n_jobs: int, optional (default=-1)
            Number of CPU cores to use for parallel computation.

        Returns:
        - dict: Cross-validation results including accuracy and the mean score.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data(df, target_column)

        cv = StratifiedKFold(n_splits=cv_splits)

        scores = cross_val_score(self.model, X_train, y_train, cv=cv, n_jobs=n_jobs)

        mean_score = np.mean(scores)
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "cross_val_scores": scores,
            "mean_cross_val_score": mean_score,
            "accuracy_on_test_set": accuracy
        }

