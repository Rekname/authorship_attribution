from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class BaseClassifier:
    name: str
    display_name: str

    def __init__(self) -> None:
        self._estimator: Any = None
        self._scaler: StandardScaler | None = None
        self.metrics_: dict[str, Any] = {}

    def _build_estimator(self):
        raise NotImplementedError

    @property
    def use_scaler(self) -> bool:
        return True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_idx: np.ndarray | None = None,
    ) -> "BaseClassifier":
        if self.use_scaler:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if test_idx is not None and len(test_idx) > 0:
            mask = np.zeros(len(y), dtype=bool)
            mask[test_idx] = True
            X_train, X_test = X[~mask], X[mask]
            y_train, y_test = y[~mask], y[mask]
        else:
            X_train, y_train = X, y
            X_test = y_test = None

        self._estimator = self._build_estimator()
        self._estimator.fit(X_train, y_train)

        self.metrics_["train_accuracy"] = float(
            accuracy_score(y_train, self._estimator.predict(X_train))
        )
        if X_test is not None:
            self.metrics_["test_accuracy"] = float(
                accuracy_score(y_test, self._estimator.predict(X_test))
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)
        scores = self._estimator.decision_function(X)
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
