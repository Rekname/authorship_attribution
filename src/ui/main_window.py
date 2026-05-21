from __future__ import annotations
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget
from src.persistence import load_training_config
from src.trainer import AuthorshipTrainer
from src.ui.widgets.analysis import AnalysisWidget
from src.ui.widgets.authors import AuthorsWidget
from src.ui.widgets.training import TrainingWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Установление авторства текста")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 820)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 0)

        self._tabs = QTabWidget()
        self.analysis = AnalysisWidget()
        self.authors = AuthorsWidget()
        self.training = TrainingWidget()
        for w, name in [(self.analysis, "Анализ"), (self.authors, "Авторы"), (self.training, "Обучение")]:
            self._tabs.addTab(w, name)
        layout.addWidget(self._tabs)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Готово.")
        self.training.model_trained.connect(self._on_model_trained)

        self._initial_load()

    def _initial_load(self) -> None:
        self.authors.refresh()
        try:
            trainer = AuthorshipTrainer.load()
            self.analysis.set_trainer(trainer)
            self._status.showMessage(f"Загружена модель: {len(trainer.authors_)} авторов.")
        except FileNotFoundError:
            pass
        except Exception as e:
            self._status.showMessage(f"Не удалось загрузить модель: {e}")

        config = load_training_config()
        if config:
            self.training.restore_from_config(config)

    def _on_model_trained(self, trainer) -> None:
        self.analysis.set_trainer(trainer)
        self.authors.refresh()
        self._status.showMessage(f"Модель обучена. Авторов: {len(trainer.authors_)}.")
