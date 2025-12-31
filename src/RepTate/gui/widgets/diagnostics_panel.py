"""Widget for displaying diagnostics."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout


class DiagnosticsPanel(QWidget):
    """Widget for displaying Bayesian inference diagnostics.

    This panel displays diagnostic information from MCMC sampling including
    convergence checks, trace plots, and summary statistics for posterior
    distributions. It provides visual feedback on sampling quality and
    parameter uncertainties.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
