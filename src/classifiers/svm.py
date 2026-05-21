from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from src.classifiers.base import BaseClassifier


class SVMClassifier(BaseClassifier):
    name = "svm"
    display_name = "SVM"

    def _build_estimator(self):
        base = LinearSVC(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            dual="auto",
            tol=1e-3,
        )
        return CalibratedClassifierCV(base, cv=3, n_jobs=-1)
