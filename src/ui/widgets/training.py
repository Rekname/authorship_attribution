from __future__ import annotations
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class _TrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            from src.loader import load_corpus
            from src.trainer import AuthorshipTrainer
            self.progress.emit(3, "Загрузка корпуса...")
            texts, labels = load_corpus()
            if len(set(labels)) < 2:
                self.error.emit("Недостаточно авторов в датасете (нужно ≥ 2).")
                return
            self.progress.emit(8, f"Загружено {len(texts)} чанков от {len(set(labels))} авторов.")
            trainer = AuthorshipTrainer()
            trainer.fit(texts, labels, progress_callback=self.progress.emit)
            trainer.save()
            self.finished.emit(trainer)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class TrainingWidget(QWidget):
    model_trained = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        self._train_btn = QPushButton("Обучить модель")
        self._train_btn.setObjectName("primary")
        self._train_btn.setMinimumHeight(42)
        self._train_btn.clicked.connect(self._on_train)
        lv.addWidget(self._train_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.hide()
        lv.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lv.addWidget(self._status)
        lv.addStretch()
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        rv.addWidget(self._summary)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Модуль", "Train accuracy", "Test accuracy"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        rv.addWidget(self._table, stretch=1)
        split.addWidget(right)
        split.setSizes([300, 760])

    def _on_train(self):
        self._train_btn.setEnabled(False)
        self._progress.show()
        self._progress.setValue(0)
        self._summary.clear()
        self._table.setRowCount(0)
        self._worker = _TrainWorker(self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self._progress.setValue(pct)
        self._status.setText(msg)

    def _on_done(self, trainer):
        self._train_btn.setEnabled(True)
        self._progress.hide()
        self._status.setText("Модель обучена и сохранена в models/.")
        metrics = {n: {**c.metrics_, "display_name": c.display_name}
                   for n, c in trainer.classifiers.items()}
        self._render(metrics, len(trainer.authors_))
        self.model_trained.emit(trainer)

    def _on_error(self, msg):
        self._train_btn.setEnabled(True)
        self._progress.hide()
        self._status.setText("Ошибка обучения.")
        QMessageBox.critical(self, "Ошибка", msg)

    def restore_from_config(self, config):
        self._render(config.get("classifier_metrics", {}), config.get("n_authors", 0))

    def _render(self, metrics, n_authors):
        self._summary.setText(f"Авторов: {n_authors}")
        self._table.setRowCount(len(metrics))
        for row, (name, m) in enumerate(metrics.items()):
            short = m.get("display_name", name).split("(")[0].strip()
            tr, te = m.get("train_accuracy"), m.get("test_accuracy")
            self._table.setItem(row, 0, QTableWidgetItem(short))
            self._table.setItem(row, 1, QTableWidgetItem(f"{tr:.1%}" if tr is not None else "—"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{te:.1%}" if te is not None else "—"))
