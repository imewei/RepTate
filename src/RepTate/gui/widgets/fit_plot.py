"""Widget for plotting deterministic fit results."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout


class FitPlot(QWidget):
    """Widget for plotting deterministic fit results.

    This panel displays the results of deterministic optimization fits,
    showing experimental data overlaid with fitted model predictions.
    It visualizes parameter estimates from least-squares optimization
    or other deterministic fitting methods.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
