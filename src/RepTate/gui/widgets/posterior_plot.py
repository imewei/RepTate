"""Widget for plotting uncertainty bands."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout


class PosteriorPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
