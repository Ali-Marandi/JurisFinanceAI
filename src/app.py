"""
JurisFinanceAI - Application Controller
Manages application lifecycle and initialization.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import sys


def create_app() -> QApplication:
    """Create and configure the QApplication."""
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("JurisFinanceAI")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Ali Marandi")

    # Default font for Persian text support
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    return app
