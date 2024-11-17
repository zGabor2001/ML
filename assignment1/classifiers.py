from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from cuml.svm import SVC as cumlSVC
from cuml.preprocessing import StandardScaler as cumlStandardScaler
from cuml.preprocessing import LabelEncoder as cumlLabelEncoder
import cupy as cp

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
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @timer
    def fit(self, X_train, y_train):
        """Fit the SVM model to the training data.

        Parameters:
        - X_train: np.ndarray
            Training features.
        - y_train: np.ndarray
            Training labels.
        """
        self.model.fit(X_train, y_train)

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
        return self.model.predict(X)

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


class GPUSupportVectorMachineClassifier:
    def __init__(self, kernel='linear', C=1.0):
        """Initialize the GPU-accelerated SVM classifier.

        Parameters:
        - kernel: str, optional (default='linear')
            Specifies the kernel type to be used in the algorithm.
        - C: float, optional (default=1.0)
            Regularization parameter.
        """
        self.kernel = kernel
        self.C = C
        self.model = cumlSVC(kernel=self.kernel, C=self.C)
        self.scaler = cumlStandardScaler()
        self.label_encoder = cumlLabelEncoder()

    @timer
    def preprocess_data(self, df, target_column):
        """Preprocess the input DataFrame.

        Parameters:
        - df: pd.DataFrame
            The input data.
        - target_column: str
            The name of the target column.

        Returns:
        - X_train, X_test, y_train, y_test: cp.ndarray
            Preprocessed training and testing data, moved to GPU memory.
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert to GPU-backed cupy arrays
        X = cp.asarray(X)
        y = cp.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardize features using GPU-based scaler
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @timer
    def fit(self, X_train, y_train):
        """Fit the SVM model to the training data.

        Parameters:
        - X_train: cp.ndarray
            Training features.
        - y_train: cp.ndarray
            Training labels.
        """
        self.model.fit(X_train, y_train)

    @timer
    def predict(self, X):
        """Predict using the trained SVM model.

        Parameters:
        - X: cp.ndarray
            Input features for prediction.

        Returns:
        - y_pred: cp.ndarray
            Predicted labels (GPU array).
        """
        return self.model.predict(X)

    @timer
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.

        Parameters:
        - X_test: cp.ndarray
            Testing features.
        - y_test: cp.ndarray
            True labels for the test set.

        Returns:
        - dict: Evaluation results containing accuracy and a classification report.
        """
        y_pred = self.predict(X_test)
        # Convert predictions to host memory (CPU) for sklearn-based metrics
        y_pred_host = cp.asnumpy(y_pred)
        y_test_host = cp.asnumpy(y_test)

        accuracy = accuracy_score(y_test_host, y_pred_host)
        report = classification_report(y_test_host, y_pred_host)
        return {
            "accuracy": accuracy,
            "classification_report": report
        }