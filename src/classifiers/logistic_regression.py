from sklearn.linear_model import LogisticRegression
from src.classifiers.base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    name = "logreg"
    display_name = "Logistic Regression"

    def _build_estimator(self):
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
        )
