"""View for configuring and displaying deterministic fit results."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class FitView(QWidget):
    """Simple placeholder view for deterministic fit results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Deterministic fit configuration"))
        layout.addWidget(QLabel("Fit results will appear here."))
