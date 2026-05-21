from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import MODEL_PATH
from src.classifiers import ALL_CLASSIFIERS, BaseClassifier
from src.features import FeatureExtractor
from src.persistence import save_training_config


class AuthorshipTrainer:
    def __init__(self, language: str = "both") -> None:
        self.language = language
        self.extractor = FeatureExtractor(language=language)
        self.label_encoder = LabelEncoder()
        self.classifiers: dict[str, BaseClassifier] = {}
        self.authors_: list[str] = []

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        progress_callback=None,
    ) -> "AuthorshipTrainer":
        def _report(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        _report(5, "Кодирование меток...")
        y = self.label_encoder.fit_transform(labels)
        self.authors_ = list(self.label_encoder.classes_)

        _report(15, "Извлечение признаков...")
        X = self.extractor.fit_transform(texts)

        try:
            _, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                stratify=y,
                random_state=42,
            )
        except ValueError:
            test_idx = np.array([], dtype=int)

        step_pct = max(1, (95 - 65) // len(ALL_CLASSIFIERS))
        pct = 65
        for cls in ALL_CLASSIFIERS:
            inst = cls()
            _report(pct, f"Обучение модуля «{inst.display_name}»...")
            inst.fit(X, y, test_idx=test_idx)
            self.classifiers[inst.name] = inst
            pct += step_pct

        _report(100, "Готово!")
        return self

    def predict_proba(self, text: str, classifier_name: str) -> dict[str, float]:
        if classifier_name not in self.classifiers:
            return {}
        X = self.extractor.transform([text])
        proba = self.classifiers[classifier_name].predict_proba(X)[0]
        clf = self.classifiers[classifier_name]._estimator
        classes = (
            clf.classes_ if hasattr(clf, "classes_") else np.arange(len(self.authors_))
        )
        return {self.authors_[int(c)]: float(p) for c, p in zip(classes, proba)}

    def ensemble_proba(self, text: str) -> dict[str, float]:
        per_clf = [self.predict_proba(text, n) for n in self.classifiers]
        per_clf = [p for p in per_clf if p]
        if not per_clf:
            return {}
        authors = set().union(*per_clf)
        avg = {a: sum(p.get(a, 0.0) for p in per_clf) / len(per_clf) for a in authors}
        total = sum(avg.values()) or 1.0
        return {a: v / total for a, v in avg.items()}

    @property
    def is_trained(self) -> bool:
        return bool(self.classifiers)

    def save(self, path: Path | None = None) -> Path:
        model_path = path or MODEL_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)
        save_training_config(
            classifier_metrics={
                name: {**clf.metrics_, "display_name": clf.display_name}
                for name, clf in self.classifiers.items()
            },
            n_authors=len(self.authors_),
        )
        return model_path

    @staticmethod
    def load(path: Path | None = None) -> "AuthorshipTrainer":
        p = path or MODEL_PATH
        if not p.exists():
            raise FileNotFoundError(f"Модель не найдена: {p}")
        return joblib.load(p)
