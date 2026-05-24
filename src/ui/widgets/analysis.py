from __future__ import annotations
import os
import re
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSplitter, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from src.ui.theme import ACCENT


class _PredictWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, trainer, text, parent=None):
        super().__init__(parent)
        self.trainer, self.text = trainer, text

    def run(self):
        try:
            self.finished.emit(self.trainer.ensemble_proba(self.text))
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class AnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._trainer = None
        self._worker = None
        self._results: dict[str, float] = {}

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        title = QLabel("Введите текст")
        title.setObjectName("h2")
        lv.addWidget(title)
        self._text = QTextEdit()
        self._text.setMinimumWidth(380)
        lv.addWidget(self._text)

        load_btn = QPushButton("Загрузить файл…")
        load_btn.clicked.connect(self._on_load_file)
        self._file_label = QLabel("")
        file_row = QHBoxLayout()
        file_row.addWidget(load_btn)
        file_row.addWidget(self._file_label, stretch=1)
        lv.addLayout(file_row)

        self._predict_btn = QPushButton("Определить автора")
        self._predict_btn.setObjectName("primary")
        self._predict_btn.setMinimumHeight(38)
        self._predict_btn.clicked.connect(self._on_predict)
        lv.addWidget(self._predict_btn)

        self._status = QLabel("")
        lv.addWidget(self._status)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self._verdict = QLabel("")
        self._verdict.setObjectName("verdict")
        self._verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rv.addWidget(self._verdict)

        self._fig = Figure(figsize=(5.5, 3.8), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        rv.addWidget(self._canvas, stretch=1)

        self._stats = QLabel("")
        self._stats.setWordWrap(True)
        rv.addWidget(self._stats)
        split.addWidget(right)
        split.setSizes([480, 600])

    def set_trainer(self, trainer):
        self._trainer = trainer
        if self._results:
            self._render()

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть текстовый файл", "",
                                              "Текстовые файлы (*.txt *.md *.rst);;Все файлы (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                self._text.setPlainText(f.read())
            self._file_label.setText(os.path.basename(path))
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _on_predict(self):
        if self._trainer is None or not self._trainer.is_trained:
            QMessageBox.warning(self, "Модель не обучена", "Сначала обучите модель на вкладке «Обучение».")
            return
        text = self._text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Нет текста", "Введите или загрузите текст.")
            return
        self._update_stats(text)

        self._predict_btn.setEnabled(False)
        self._status.setText("Анализирую…")
        self._verdict.setText("")
        self._worker = _PredictWorker(self._trainer, text, self)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, proba):
        self._predict_btn.setEnabled(True)
        self._status.setText("")
        self._results = proba
        self._render()

    def _on_error(self, msg):
        self._predict_btn.setEnabled(True)
        self._status.setText("Ошибка анализа.")
        QMessageBox.critical(self, "Ошибка", msg)

    def _render(self):
        if not self._results:
            self._verdict.setText("Нет данных")
            return
        top = max(self._results, key=self._results.get)
        self._verdict.setText(f"{top} — {self._results[top]:.1%}")
        items = sorted(self._results.items(), key=lambda x: x[1])[-12:]
        self._ax.clear()
        bars = self._ax.barh([n for n, _ in items], [v for _, v in items], color=ACCENT, alpha=0.85)
        self._ax.set_xlim(0, 1)
        self._ax.set_xlabel("Вероятность")
        for bar, val in zip(bars, [v for _, v in items]):
            self._ax.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                          f"{val:.1%}", va="center", fontsize=9)
        self._canvas.draw()

    def _update_stats(self, text):
        words = re.findall(r"\b\w+\b", text)
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        avg = sum(len(re.findall(r"\b\w+\b", s)) for s in sents) / max(len(sents), 1)
        ttr = len(set(w.lower() for w in words)) / max(len(words), 1)
        self._stats.setText(f"Слов: {len(words):,}   Предложений: {len(sents)}   "
                            f"Ср. длина: {avg:.1f}   TTR: {ttr:.3f}")
