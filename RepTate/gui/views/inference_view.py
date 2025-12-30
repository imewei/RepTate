"""View for Bayesian inference configuration and results."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class InferenceView(QWidget):
    """Simple placeholder for inference output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Bayesian inference configuration"))
        layout.addWidget(QLabel("Posterior summaries will appear here."))
