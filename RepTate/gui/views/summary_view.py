"""Summary view for residuals and posterior statistics."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SummaryView(QWidget):
    """Placeholder summary view for numerical diagnostics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Residual diagnostics summary"))
        layout.addWidget(QLabel("Posterior statistics summary"))
