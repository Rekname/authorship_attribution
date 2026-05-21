from __future__ import annotations

BG, SURFACE, INPUT, BORDER = "#121212", "#1e1e1e", "#252525", "#3a3a3a"
TEXT, MUTED = "#f0f0f0", "#707070"
ACCENT, ACCENT_HOVER, ACCENT_DARK = "#5294e2", "#6ba4ea", "#3a7bd0"
OK_GREEN, ERROR = "#73d216", "#ef2929"

CHART_COLORS = [ACCENT, OK_GREEN, "#ff9800", "#ad7fa8", "#00bcd4", ERROR]

QSS = f"""
* {{ color: {TEXT}; font-family: "Segoe UI", "Ubuntu", sans-serif; font-size: 13px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ background: {SURFACE}; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; padding: 4px 8px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {SURFACE}; border-radius: 4px; }}
QTabBar::tab {{ background: {BG}; padding: 9px 22px; border: 1px solid {BORDER}; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
QTabBar::tab:selected {{ background: {SURFACE}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton {{ background: #2a2a2a; border: 1px solid #4a4a4a; padding: 8px 16px; border-radius: 4px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: none; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_DARK}; }}
QLineEdit, QTextEdit, QComboBox {{ background: {INPUT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px; selection-background-color: {ACCENT}; }}
QTextEdit:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: #2a2a2a; selection-background-color: {ACCENT}; }}
QProgressBar {{ background: {INPUT}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QTableWidget {{ background: {SURFACE}; alternate-background-color: {INPUT}; gridline-color: {BORDER}; border: 1px solid {BORDER}; border-radius: 4px; selection-background-color: {ACCENT}; }}
QHeaderView::section {{ background: #2a2a2a; border: none; border-right: 1px solid {BORDER}; padding: 6px; }}
QScrollBar:vertical {{ background: {BG}; width: 12px; }}
QScrollBar::handle:vertical {{ background: #4a4a4a; border-radius: 6px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QLabel#h2 {{ font-size: 15px; font-weight: 500; }}
QLabel#verdict {{ font-size: 18px; font-weight: 600; color: {ACCENT}; padding: 10px; }}
"""


def apply_dark_theme(app) -> None:
    import matplotlib

    matplotlib.use("QtAgg", force=True)
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#4a4a4a",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": BORDER,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "legend.facecolor": "#2a2a2a",
            "legend.edgecolor": BORDER,
            "legend.labelcolor": TEXT,
            "axes.prop_cycle": cycler(color=CHART_COLORS),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    app.setStyleSheet(QSS)
