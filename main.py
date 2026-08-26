"""
JurisFinanceAI - Commercial Legal & Finance AI Desktop Application
Main entry point.

Version: 1.0.0
Author: Ali Marandi

A professional-grade AI-powered legal and financial assistant
for Windows, featuring document analysis, contract review,
financial analysis, and risk assessment.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import create_app
from src.ui.main_window import MainWindow
from src.core.database import get_database


def main():
    # Create application
    app = create_app()

    # Initialize database
    db = get_database()

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    exit_code = app.exec()

    # Cleanup
    db.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
