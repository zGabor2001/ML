import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def select_variables(X, y):
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    # Beispiel für ein lineares SVM-Modell
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=5, step=1)
    selector = selector.fit(X, y)
    print(selector.support_)


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

        #Encode target if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)

        # Handle categorical features if present
        X = pd.get_dummies(X, drop_first=True)

        select_variables(X, y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        """Fit the SVM model to the training data.

        Parameters:
        - X_train: np.ndarray
            Training features.
        - y_train: np.ndarray
            Training labels.
        """
        self.model.fit(X_train, y_train)

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
