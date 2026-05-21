from src.classifiers.base import BaseClassifier
from src.classifiers.svm import SVMClassifier
from src.classifiers.random_forest import RandomForestAuthorshipClassifier
from src.classifiers.logistic_regression import LogisticRegressionClassifier
ALL_CLASSIFIERS: list[type[BaseClassifier]] = [
    SVMClassifier,
    RandomForestAuthorshipClassifier,
    LogisticRegressionClassifier,
]
