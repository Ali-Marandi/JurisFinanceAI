from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QLabel, QPushButton, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from ..core.database import get_database


class StatCard(QFrame):
    """A statistics card widget."""
    def __init__(self, title: str, value: str, icon: str, color: str = "#3b82f6", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedHeight(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        # Icon container
        icon_label = QLabel(icon)
        icon_label.setFixedSize(48, 48)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color}20;
                border-radius: 14px;
                font-size: 22px;
            }}
        """)
        layout.addWidget(icon_label)

        # Text content
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {color}; background: transparent;")
        text_layout.addWidget(value_label)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        text_layout.addWidget(title_label)

        text_layout.addStretch()
        layout.addLayout(text_layout)
        layout.addStretch()


class ActivityItem(QFrame):
    """Recent activity item."""
    def __init__(self, icon: str, title: str, description: str, time: str, color: str = "#3b82f6", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color}; font-size: 10px; background: transparent;")
        layout.addWidget(dot)

        info = QVBoxLayout()
        info.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("font-size: 13px; font-weight: 500; color: #f1f5f9; background: transparent;")
        info.addWidget(t)
        d = QLabel(description)
        d.setStyleSheet("font-size: 12px; color: #64748b; background: transparent;")
        info.addWidget(d)
        layout.addLayout(info)
        layout.addStretch()

        time_label = QLabel(time)
        time_label.setStyleSheet("font-size: 11px; color: #475569; background: transparent;")
        layout.addWidget(time_label)


class DashboardPage(QWidget):
    """Dashboard overview page."""

    navigate_request = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.db = get_database()
        self._init_ui()
        # Refresh data periodically
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh_stats_only)
        self._refresh_timer.start(30000)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.stat_conversations = StatCard("گفتگوها", "۰", "💬", "#3b82f6")
        self.stat_documents = StatCard("اسناد", "۰", "📄", "#8b5cf6")
        self.stat_cases = StatCard("پرونده‌ها", "۰", "📁", "#22c55e")
        self.stat_risks = StatCard("ارزیابی ریسک", "۰", "⚠", "#f59e0b")

        for card in [self.stat_conversations, self.stat_documents, self.stat_cases, self.stat_risks]:
            stats_layout.addWidget(card)

        main_layout.addLayout(stats_layout)

        # Charts and Activity row
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(16)

        # Chart card
        chart_card = QFrame()
        chart_card.setObjectName("card")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 16, 20, 16)

        chart_title = QLabel("نمودار فعالیت‌ها")
        chart_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent; padding-bottom: 12px;")
        chart_layout.addWidget(chart_title)

        self.figure = Figure(figsize=(6, 3), facecolor='#1e2a45')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(260)
        chart_layout.addWidget(self.canvas)

        bottom_layout.addWidget(chart_card, 2)

        # Recent activity card
        activity_card = QFrame()
        activity_card.setObjectName("card")
        activity_layout = QVBoxLayout(activity_card)
        activity_layout.setContentsMargins(20, 16, 20, 16)

        act_title = QLabel("فعالیت‌های اخیر")
        act_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent; padding-bottom: 12px;")
        activity_layout.addWidget(act_title)

        self.activity_list = QVBoxLayout()
        self.activity_list.setSpacing(4)

        sample_activities = [
            ("💬", "گفتگوی جدید", "تحلیل قرارداد اجاره", "۵ دقیقه پیش", "#3b82f6"),
            ("📄", "سند جدید", "قرارداد همکاری آپلود شد", "۲ ساعت پیش", "#8b5cf6"),
            ("💰", "ثبت تراکنش", "هزینه مشاوره حقوقی", "۳ ساعت پیش", "#22c55e"),
            ("⚠", "ارزیابی ریسک", "بررسی ریسک قرارداد", "دیروز", "#f59e0b"),
            ("📁", "پرونده جدید", "پرونده دعوای ملکی", "دیروز", "#06b6d4"),
        ]

        for icon, title, desc, time, color in sample_activities:
            item = ActivityItem(icon, title, desc, time, color)
            self.activity_list.addWidget(item)

        self.activity_list.addStretch()
        activity_layout.addLayout(self.activity_list)

        bottom_layout.addWidget(activity_card, 1)

        main_layout.addLayout(bottom_layout, 1)

        # Quick actions
        quick_card = QFrame()
        quick_card.setObjectName("card")
        quick_layout = QHBoxLayout(quick_card)
        quick_layout.setContentsMargins(20, 16, 20, 16)

        quick_title = QLabel("دسترسی سریع")
        quick_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent;")
        quick_layout.addWidget(quick_title)
        quick_layout.addSpacing(20)

        actions = [
            ("💬 گفتگوی جدید", "#3b82f6"),
            ("📄 آپلود سند", "#8b5cf6"),
            ("🛡 تحلیل قرارداد", "#22c55e"),
            ("💰 ثبت تراکنش", "#f59e0b"),
        ]
        for text, color in actions:
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color}20;
                    color: {color};
                    border: 1px solid {color}40;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 13px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {color}30;
                    border-color: {color};
                }}
            """)
            btn.clicked.connect(lambda checked, t=text: self._on_quick_action(t))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        main_layout.addWidget(quick_card)

        self._refresh_data()

    def _refresh_data(self):
        """Refresh all dashboard data from database."""
        self._refresh_stats_only()
        self._draw_chart()

    def _refresh_stats_only(self):
        """Refresh only the stat cards without redrawing the chart."""
        try:
            analytics = self.db.get_analytics_summary()
            self.stat_conversations.findChild(QLabel).setText(str(analytics["total_conversations"]))
            self.stat_documents.findChild(QLabel).setText(str(analytics["total_documents"]))
            self.stat_cases.findChild(QLabel).setText(str(analytics["total_cases"]))
        except Exception:
            pass

    def _on_quick_action(self, text: str):
        """Handle quick action button clicks."""
        action_map = {
            "💬 گفتگوی جدید": "chat",
            "📄 آپلود سند": "documents",
            "🛡 تحلیل قرارداد": "chat",
            "💰 ثبت تراکنش": "finance",
        }
        page = action_map.get(text, "chat")
        self.navigate_request.emit(page)

    def _draw_chart(self):
        """Draw the activity chart."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1e2a45')

        # NOTE: Hardcoded placeholder data — replace with DB-driven queries when available
        days = ['شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه']
        conversations = np.array([5, 8, 12, 7, 15, 10, 3])
        documents = np.array([2, 4, 6, 3, 8, 5, 1])

        x = np.arange(len(days))
        width = 0.35

        bars1 = ax.bar(x - width/2, conversations, width, label='گفتگوها', color='#3b82f6', alpha=0.8)
        bars2 = ax.bar(x + width/2, documents, width, label='اسناد', color='#8b5cf6', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(days, color='#94a3b8', fontsize=10)
        ax.tick_params(axis='y', colors='#64748b', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.legend(facecolor='#1e2a45', edgecolor='#334155', labelcolor='#94a3b8', fontsize=10)
        ax.grid(axis='y', color='#334155', alpha=0.5, linestyle='--')

        self.figure.tight_layout()
        self.canvas.draw()
