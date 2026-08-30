from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QDoubleSpinBox, QTextEdit, QDateEdit, QAbstractItemView,
    QDialog, QFormLayout, QDialogButtonBox, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QColor
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import numpy as np

from ..core.database import get_database


class AddRecordDialog(QDialog):
    """Dialog for adding a financial record."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ثبت تراکنش جدید")
        self.setFixedSize(400, 300)
        self.setStyleSheet("background-color: #1a1a2e; color: #f1f5f9;")

        layout = QFormLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["income", "expense", "settlement", "fee"])
        type_labels = {"income": "درآمد", "expense": "هزینه", "settlement": "توافق", "fee": "حق‌الوکاله"}
        self.type_combo.setCurrentText("income")
        layout.addRow("نوع تراکنش:", self.type_combo)

        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0, 9999999999)
        self.amount_spin.setDecimals(0)
        self.amount_spin.setSuffix(" ریال")
        self.amount_spin.setStyleSheet("background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155; border-radius: 8px; padding: 8px;")
        layout.addRow("مبلغ:", self.amount_spin)

        self.desc_input = QTextEdit()
        self.desc_input.setMaximumHeight(60)
        self.desc_input.setStyleSheet("background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155; border-radius: 8px; padding: 8px;")
        layout.addRow("توضیحات:", self.desc_input)

        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy/MM/dd")
        self.date_edit.setStyleSheet("background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155; border-radius: 8px; padding: 8px;")
        layout.addRow("تاریخ:", self.date_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setStyleSheet("""
            QPushButton { background-color: #3b82f6; color: white; border: none; border-radius: 8px; padding: 8px 20px; }
            QPushButton:disabled { background-color: #334155; }
        """)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)


class FinancePage(QWidget):
    """Financial analysis and records page."""

    def __init__(self):
        super().__init__()
        self.db = get_database()
        self._init_ui()
        self._refresh_data()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Summary cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        self.card_income = self._make_stat_card("درآمد کل", "۰", "#22c55e")
        self.card_expense = self._make_stat_card("هزینه‌ها", "۰", "#ef4444")
        self.card_balance = self._make_stat_card("موجودی", "۰", "#3b82f6")
        self.card_cases = self._make_stat_card("پرونده‌های فعال", "۰", "#8b5cf6")

        for card in [self.card_income, self.card_expense, self.card_balance, self.card_cases]:
            cards_layout.addWidget(card)

        main_layout.addLayout(cards_layout)

        # Charts + Table
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Chart
        chart_card = QFrame()
        chart_card.setObjectName("card")
        chart_lay = QVBoxLayout(chart_card)
        chart_lay.setContentsMargins(16, 16, 16, 16)

        chart_title = QLabel("نمودار مالی")
        chart_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #f8fafc; background: transparent;")
        chart_lay.addWidget(chart_title)

        self.figure = Figure(figsize=(5, 3), facecolor='#1e2a45')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        chart_lay.addWidget(self.canvas)

        content_layout.addWidget(chart_card, 1)

        # Records table
        table_card = QFrame()
        table_card.setObjectName("card")
        table_lay = QVBoxLayout(table_card)
        table_lay.setContentsMargins(16, 16, 16, 16)

        table_header = QHBoxLayout()
        table_title = QLabel("تراکنش‌ها")
        table_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #f8fafc; background: transparent;")
        table_header.addWidget(table_title)
        table_header.addStretch()

        add_btn = QPushButton("+ ثبت تراکنش")
        add_btn.setFixedHeight(36)
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet("""
            QPushButton { background-color: #22c55e; color: white; border: none; border-radius: 8px; font-size: 12px; font-weight: 600; }
            QPushButton:hover { background-color: #16a34a; }
        """)
        add_btn.clicked.connect(self._add_record)
        table_header.addWidget(add_btn)
        table_lay.addLayout(table_header)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["تاریخ", "نوع", "مبلغ (ریال)", "توضیحات"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1e2a45; border: none; gridline-color: #334155; color: #e2e8f0; }
            QTableWidget::item:selected { background-color: #1a3a6e; }
            QHeaderView::section { background-color: #16213e; color: #94a3b8; font-weight: 600; border: none; padding: 8px; }
        """)
        table_lay.addWidget(self.table)

        content_layout.addWidget(table_card, 1)
        main_layout.addLayout(content_layout, 1)

    def _make_stat_card(self, label: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setFixedHeight(100)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 12, 20, 12)

        val_label = QLabel(value)
        val_label.setStyleSheet(f"font-size: 24px; font-weight: 700; color: {color}; background: transparent;")
        val_label.setObjectName("stat_value")
        layout.addWidget(val_label)

        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 12px; color: #94a3b8; background: transparent;")
        lbl.setObjectName("stat_label")
        layout.addWidget(lbl)

        return card

    def _refresh_data(self):
        try:
            summary = self.db.get_financial_summary()
            self._format_number(self.card_income, summary["income"])
            self._format_number(self.card_expense, summary["expense"])
            self._format_number(self.card_balance, summary["balance"])

            analytics = self.db.get_analytics_summary()
            for lbl in self.card_cases.findChildren(QLabel):
                if lbl.objectName() == "stat_value":
                    lbl.setText(str(analytics.get("active_cases", 0)))

            records = self.db.get_financial_records()
            self.table.setRowCount(len(records))
            type_map = {"income": "درآمد", "expense": "هزینه", "settlement": "توافق", "fee": "حق‌الوکاله"}
            type_colors = {"income": "#22c55e", "expense": "#ef4444", "settlement": "#3b82f6", "fee": "#f59e0b"}

            for i, rec in enumerate(records):
                rtype = rec["record_type"]
                items = [
                    rec["date"],
                    type_map.get(rtype, rtype),
                    f"{rec['amount']:,.0f}",
                    rec.get("description", "") or "-",
                ]
                for j, text in enumerate(items):
                    item = QTableWidgetItem(str(text))
                    item.setForeground(QColor(type_colors.get(rtype, "#e2e8f0") if j == 1 else QColor("#e2e8f0")))
                    self.table.setItem(i, j, item)

            self._draw_chart()
        except Exception:
            pass

    def _format_number(self, card: QFrame, value: float):
        for lbl in card.findChildren(QLabel):
            if lbl.objectName() == "stat_value":
                lbl.setText(f"{value:,.0f}")

    def _add_record(self):
        dialog = AddRecordDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            rtype = dialog.type_combo.currentText()
            amount = dialog.amount_spin.value()
            desc = dialog.desc_input.toPlainText()
            date = dialog.date_edit.date().toString("yyyy-MM-dd")
            self.db.add_financial_record(rtype, amount, desc, date)
            self._refresh_data()

    def _draw_chart(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1e2a45')

        # NOTE: Hardcoded placeholder data — replace with DB-driven queries when available
        months = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور']
        income = np.array([45000, 52000, 38000, 61000, 48000, 55000])
        expense = np.array([32000, 28000, 35000, 42000, 30000, 38000])

        x = np.arange(len(months))
        ax.plot(x, income, color='#22c55e', marker='o', linewidth=2, markersize=6, label='درآمد')
        ax.plot(x, expense, color='#ef4444', marker='s', linewidth=2, markersize=6, label='هزینه')
        ax.fill_between(x, income, alpha=0.1, color='#22c55e')
        ax.fill_between(x, expense, alpha=0.1, color='#ef4444')

        ax.set_xticks(x)
        ax.set_xticklabels(months, color='#94a3b8', fontsize=10)
        ax.tick_params(axis='y', colors='#64748b', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.legend(facecolor='#1e2a45', edgecolor='#334155', labelcolor='#94a3b8', fontsize=10)
        ax.grid(axis='y', color='#334155', alpha=0.5, linestyle='--')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

        self.figure.tight_layout()
        self.canvas.draw()
