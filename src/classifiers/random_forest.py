from sklearn.ensemble import RandomForestClassifier
from src.classifiers.base import BaseClassifier


class RandomForestAuthorshipClassifier(BaseClassifier):
    name = "random_forest"
    display_name = "Random Forest"

    @property
    def use_scaler(self) -> bool:
        return False

    def _build_estimator(self):
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=2,
            class_weight="balanced",
            n_jobs=-1,
        )
