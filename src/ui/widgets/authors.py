from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from src.ui.theme import ACCENT, OK_GREEN, ERROR


ROWS = [
    ("yules_k", "Yule's K"),
    ("total_sentences", "Предложений"),
    ("total_words", "Слов"),
    ("unique_words", "Уникальных слов"),
    ("punctuation_density", "Доля пунктуации"),
    ("avg_word_length", "Средняя длина слова (букв)"),
    ("avg_sentence_length_chars", "Средняя длина предложения (симв.)"),
]


class _StatsWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, author, parent=None):
        super().__init__(parent)
        self.author = author

    def run(self):
        try:
            from src.loader import get_author_statistics
            self.finished.emit(get_author_statistics(self.author))
        except Exception:
            self.finished.emit({})


class AuthorsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        lbl = QLabel("Автор:")
        lbl.setObjectName("h2")
        top.addWidget(lbl)
        self._combo = QComboBox()
        self._combo.setMinimumWidth(260)
        self._combo.currentTextChanged.connect(self._on_changed)
        top.addWidget(self._combo, stretch=1)
        root.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        split.addWidget(self._table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)
        self._ax_s, self._cv_s = self._mk_chart(v)
        self._ax_w, self._cv_w = self._mk_chart(v)
        v.addStretch()
        scroll.setWidget(inner)
        split.addWidget(scroll)
        split.setSizes([380, 600])
        root.addWidget(split, stretch=1)

    @staticmethod
    def _mk_chart(layout):
        fig = Figure(figsize=(5, 2.8), tight_layout=True)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        return ax, canvas

    def refresh(self):
        try:
            from src.loader import get_all_disk_authors
            authors = [n for n, _ in get_all_disk_authors()]
        except Exception:
            authors = []
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(authors)
        self._combo.blockSignals(False)
        if authors:
            self._on_changed(authors[0])
        else:
            self._table.setRowCount(0)

    def _on_changed(self, author):
        if not author:
            return
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._worker = _StatsWorker(author, self)
        self._worker.finished.connect(self._on_stats)
        self._worker.start()

    def _on_stats(self, stats):
        if not stats:
            return
        rows = [(label, stats[k]) for k, label in ROWS if k in stats]
        self._table.setRowCount(len(rows))
        for i, (label, val) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(label))
            if isinstance(val, float):
                disp = f"{val:,.4f}" if val < 1 else f"{val:,.2f}"
            else:
                disp = f"{val:,}" if isinstance(val, int) else str(val)
            self._table.setItem(i, 1, QTableWidgetItem(disp))
        self._table.resizeColumnsToContents()

        self._draw(self._ax_s, self._cv_s, stats.get("_sentence_lengths", []),
                   "Слов в предложении", ACCENT, "hist")
        self._draw(self._ax_w, self._cv_w, stats.get("_word_lengths", []),
                   "Букв в слове", OK_GREEN, "bar")

    @staticmethod
    def _draw(ax, canvas, data, xlabel, color, kind):
        ax.clear()
        if not data:
            canvas.draw()
            return
        arr = np.array(data)
        if kind == "hist":
            bins = min(40, max(8, int(np.sqrt(len(arr)))))
            ax.hist(arr[arr < np.percentile(arr, 98)], bins=bins, color=color, alpha=0.85)
        else:
            mx = min(20, int(arr.max()))
            lengths = np.arange(1, mx + 1)
            ax.bar(lengths, [(arr == l).sum() for l in lengths], color=color, alpha=0.85)
        m = float(np.mean(arr))
        ax.axvline(m, color=ERROR, linestyle="--", label=f"μ = {m:.1f}")
        ax.legend(loc="upper right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Количество")
        canvas.draw()
