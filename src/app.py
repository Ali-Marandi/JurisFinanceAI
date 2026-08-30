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
    from src import __version__
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Ali Marandi")

    # Default font for Persian text support
    try:
        from src.core.config import get_config
        config = get_config()
        font_size = config.get('ui.font_size', 10)
        font = QFont('Segoe UI', int(font_size))
    except Exception:
        font = QFont('Segoe UI', 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    return app
