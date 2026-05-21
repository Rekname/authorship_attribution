from __future__ import annotations
import sys
from PyQt6.QtWidgets import QApplication
from src.ui.theme import apply_dark_theme
from src.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
