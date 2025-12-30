"""Plot view placeholders for data, fit, and posterior curves."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlotViews(QWidget):
    """Container for plot placeholders."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Raw data plot placeholder"))
        layout.addWidget(QLabel("Fitted curve plot placeholder"))
        layout.addWidget(QLabel("Posterior summary plot placeholder"))
