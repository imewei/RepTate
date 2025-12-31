"""Widget for plotting uncertainty bands."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout


class PosteriorPlot(QWidget):
    """Widget for plotting uncertainty bands from Bayesian inference.

    This panel displays posterior predictive distributions and uncertainty
    quantification from MCMC sampling. It shows credible intervals and
    uncertainty bands around model predictions, allowing visual assessment
    of parameter uncertainty propagation.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
