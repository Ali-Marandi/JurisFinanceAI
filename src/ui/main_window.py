"""
JurisFinanceAI - Main Window
Professional main application window with sidebar navigation.
"""

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QFrame, QLabel, QPushButton, QScrollArea, QSizePolicy, QStatusBar
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from .themes import get_stylesheet, DARK_THEME, LIGHT_THEME
from .dashboard import DashboardPage
from .ai_chat import AIChatPage
from .documents import DocumentsPage
from .contracts import ContractsPage
from .finance import FinancePage
from .risk import RiskPage
from .settings import SettingsPage
from .quant_dashboard import QuantDashboard
from ..core.config import get_config
from ..core.database import get_database


class MainWindow(QMainWindow):
    """Main application window for JurisFinanceAI."""

    PAGE_TITLES = {
        "dashboard": "داشبورد",
        "chat": "دستیار هوشمند حقوقی و مالی",
        "documents": "مدیریت اسناد",
        "contracts": "تحلیل قراردادها",
        "finance": "تحلیل مالی",
        "risk": "ارزیابی ریسک",
        "quant": "تحلیل فایننس کمی",
        "settings": "تنظیمات",
    }

    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.db = get_database()
        self.current_theme = self.config.get("ui.theme", "dark")

        self.setWindowTitle("JurisFinanceAI - دستیار هوش مصنوعی حقوقی و مالی")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)

        self._apply_theme()
        self._init_ui()
        self._navigate_to("dashboard")

    def _apply_theme(self):
        self.setStyleSheet(get_stylesheet(self.current_theme))

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = self._create_sidebar()
        main_layout.addWidget(self.sidebar)

        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.toolbar = self._create_toolbar()
        content_layout.addWidget(self.toolbar)

        self.stacked_pages = QStackedWidget()
        self._init_pages()
        content_layout.addWidget(self.stacked_pages)

        main_layout.addWidget(content_container, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("JurisFinanceAI آماده به کار است")

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(4)

        # Logo
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(8, 4, 8, 20)
        logo_layout.setSpacing(10)
        logo_icon = QLabel("JF")
        logo_icon.setFixedSize(36, 36)
        logo_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_icon.setStyleSheet("background-color: #3b82f6; border-radius: 10px; font-size: 18px; color: white; font-weight: bold;")
        logo_layout.addWidget(logo_icon)
        app_name = QLabel("JurisFinanceAI")
        app_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #f1f5f9; background: transparent;")
        logo_layout.addWidget(app_name)
        logo_layout.addStretch()
        layout.addWidget(logo_container)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #1e293b;")
        layout.addWidget(sep)

        # Nav items: (key, icon_char, label)
        nav_items = [
            ("dashboard", "◉", "داشبورد"),
            ("chat", "💬", "دستیار هوشمند"),
            ("documents", "📄", "مدیریت اسناد"),
            ("contracts", "🛡", "تحلیل قراردادها"),
            ("finance", "💰", "تحلیل مالی"),
            ("risk", "⚠", "ارزیابی ریسک"),
            ("quant", "📐", "فایننس کمی"),
        ]

        self.nav_buttons = {}
        for key, icon, label in nav_items:
            btn = QPushButton(f"  {icon}   {label}")
            btn.setFixedHeight(44)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 10px;
                    padding: 0px 16px;
                    text-align: left;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #1a2744;
                    color: #e2e8f0;
                }
            """)
            btn.clicked.connect(lambda checked, k=key: self._navigate_to(k))
            self.nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: #1e293b;")
        layout.addWidget(sep2)

        settings_btn = QPushButton(f"  ⚙   تنظیمات")
        settings_btn.setFixedHeight(44)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 10px;
                padding: 0px 16px;
                text-align: left;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1a2744;
                color: #e2e8f0;
            }
        """)
        settings_btn.clicked.connect(lambda: self._navigate_to("settings"))
        self.nav_buttons["settings"] = settings_btn
        layout.addWidget(settings_btn)

        version = QLabel("نسخه ۴.۰.۰")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #475569; font-size: 11px; background: transparent; padding: 8px;")
        layout.addWidget(version)

        return sidebar

    def _create_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        toolbar.setFixedHeight(56)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(24, 0, 24, 0)

        self.page_title = QLabel("داشبورد")
        self.page_title.setStyleSheet("font-size: 20px; font-weight: 700; background: transparent; color: #f8fafc;")
        layout.addWidget(self.page_title)
        layout.addStretch()

        self.theme_btn = QPushButton("🌙" if self.current_theme == "dark" else "☀")
        self.theme_btn.setFixedSize(40, 40)
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.setToolTip("تغییر تم")
        self.theme_btn.setStyleSheet("""
            QPushButton { background-color: transparent; border: 1px solid #334155; border-radius: 10px; font-size: 18px; }
            QPushButton:hover { border-color: #3b82f6; }
        """)
        self.theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_btn)

        return toolbar

    def _init_pages(self):
        self.dashboard_page = DashboardPage()
        self.chat_page = AIChatPage()
        self.documents_page = DocumentsPage()
        self.contracts_page = ContractsPage()
        self.finance_page = FinancePage()
        self.risk_page = RiskPage()
        self.quant_page = QuantDashboard()
        self.settings_page = SettingsPage(self)

        self.pages = {
            "dashboard": self.dashboard_page,
            "chat": self.chat_page,
            "documents": self.documents_page,
            "contracts": self.contracts_page,
            "finance": self.finance_page,
            "risk": self.risk_page,
            "quant": self.quant_page,
            "settings": self.settings_page,
        }

        for page in self.pages.values():
            self.stacked_pages.addWidget(page)

    def _navigate_to(self, page_name: str):
        page = self.pages.get(page_name)
        if page:
            self.stacked_pages.setCurrentWidget(page)
            self.page_title.setText(self.PAGE_TITLES.get(page_name, page_name))
            # Update sidebar active state
            for key, btn in self.nav_buttons.items():
                if key == page_name:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #1a3a6e;
                            color: #f1f5f9;
                            border: none;
                            border-radius: 10px;
                            padding: 0px 16px;
                            text-align: left;
                            font-size: 13px;
                            font-weight: 600;
                            border-right: 3px solid #3b82f6;
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            color: #94a3b8;
                            border: none;
                            border-radius: 10px;
                            padding: 0px 16px;
                            text-align: left;
                            font-size: 13px;
                        }
                        QPushButton:hover {
                            background-color: #1a2744;
                            color: #e2e8f0;
                        }
                    """)
            self.db.log_action(f"navigate_{page_name}")
            self.status_bar.showMessage(f"{self.PAGE_TITLES.get(page_name, page_name)}")

    def _toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.config.set("ui.theme", self.current_theme)
        self._apply_theme()
        self.theme_btn.setText("🌙" if self.current_theme == "dark" else "☀")
        # Re-apply sidebar colors
        self._navigate_to(list(self.pages.keys())[self.stacked_pages.currentIndex()])

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def refresh_theme(self):
        """Public method to refresh theme after settings change."""
        self.current_theme = self.config.get("ui.theme", "dark")
        self._apply_theme()
        self.theme_btn.setText("🌙" if self.current_theme == "dark" else "☀")
        self._navigate_to(list(self.pages.keys())[self.stacked_pages.currentIndex()])

    def closeEvent(self, event):
        self.db.close()
        event.accept()
