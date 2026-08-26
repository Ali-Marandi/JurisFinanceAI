import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QTextEdit, QComboBox, QProgressBar, QListWidget,
    QListWidgetItem, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.database import get_database
from ..core.ai_engine import get_ai_engine


class RiskWorker(QThread):
    """Worker thread for risk assessment."""
    assessment_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, description: str, context: str):
        super().__init__()
        self.description = description
        self.context = context

    def run(self):
        try:
            ai = get_ai_engine()
            result = ai.assess_risk(self.description, self.context)
            self.assessment_done.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class RiskPage(QWidget):
    """Risk assessment and management page."""

    def __init__(self):
        super().__init__()
        self.db = get_database()
        self._init_ui()
        self._load_assessments()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("ارزیابی ریسک")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f8fafc; background: transparent;")
        header.addWidget(title)
        header.addStretch()

        new_btn = QPushButton("+ ارزیابی جدید")
        new_btn.setFixedHeight(40)
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.setStyleSheet("""
            QPushButton { background-color: #f59e0b; color: white; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; }
            QPushButton:hover { background-color: #d97706; }
        """)
        new_btn.clicked.connect(self._show_new_assessment)
        header.addWidget(new_btn)
        main_layout.addLayout(header)

        # Content
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Assessment form
        form_card = QFrame()
        form_card.setObjectName("card")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(12)

        form_title = QLabel("ارزیابی جدید")
        form_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent;")
        form_layout.addWidget(form_title)

        desc_label = QLabel("شرح موضوع ارزیابی")
        desc_label.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        form_layout.addWidget(desc_label)

        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("موضوعی که نیاز به ارزیابی ریسک دارد را شرح دهید...\nمثال: سرمایه‌گذاری در بازار بورس با مبلغ ۵۰۰ میلیون ریال")
        self.desc_input.setMaximumHeight(150)
        self.desc_input.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; font-size: 13px; }
            QTextEdit:focus { border-color: #f59e0b; }
        """)
        form_layout.addWidget(self.desc_input)

        ctx_label = QLabel("زمینه و اطلاعات تکمیلی (اختیاری)")
        ctx_label.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        form_layout.addWidget(ctx_label)

        self.context_input = QTextEdit()
        self.context_input.setPlaceholderText("اطلاعات تکمیلی، محدودیت‌ها، شرایط...")
        self.context_input.setMaximumHeight(100)
        self.context_input.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; font-size: 13px; }
        """)
        form_layout.addWidget(self.context_input)

        self.assess_btn = QPushButton("شروع ارزیابی ریسک")
        self.assess_btn.setFixedHeight(44)
        self.assess_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.assess_btn.setStyleSheet("""
            QPushButton { background-color: #f59e0b; color: white; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; }
            QPushButton:hover { background-color: #d97706; }
        """)
        self.assess_btn.clicked.connect(self._start_assessment)
        form_layout.addWidget(self.assess_btn)

        content_layout.addWidget(form_card, 1)

        # Results panel
        results_card = QFrame()
        results_card.setObjectName("card")
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.setSpacing(12)

        results_title = QLabel("نتایج ارزیابی")
        results_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent;")
        results_layout.addWidget(results_title)

        # Risk score display
        score_frame = QFrame()
        score_layout = QVBoxLayout(score_frame)
        score_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.setSpacing(8)

        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("font-size: 48px; font-weight: 700; color: #f59e0b; background: transparent;")
        score_layout.addWidget(self.score_label)

        self.level_label = QLabel("")
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #94a3b8; background: transparent;")
        score_layout.addWidget(self.level_label)

        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(False)
        self.score_bar.setFixedHeight(8)
        score_layout.addWidget(self.score_bar)

        results_layout.addWidget(score_frame)

        self.results_content = QTextEdit()
        self.results_content.setReadOnly(True)
        self.results_content.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; font-size: 13px; }
        """)
        self.results_content.setHtml("<div style='text-align: center; color: #64748b; padding: 40px;'>نتایج ارزیابی اینجا نمایش داده می‌شود</div>")
        results_layout.addWidget(self.results_content)

        content_layout.addWidget(results_card, 1)
        main_layout.addLayout(content_layout, 1)

        # History section
        history_label = QLabel("تاریخچه ارزیابی‌ها")
        history_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #f8fafc; background: transparent;")
        main_layout.addWidget(history_label)

        self.history_list = QListWidget()
        self.history_list.setFixedHeight(160)
        self.history_list.setStyleSheet("""
            QListWidget { background-color: transparent; border: 1px solid #334155; border-radius: 10px; color: #e2e8f0; outline: none; }
            QListWidget::item { padding: 10px 16px; margin: 2px; border-radius: 6px; }
            QListWidget::item:hover { background-color: #243352; }
            QListWidget::item:selected { background-color: #1a3a6e; }
        """)
        main_layout.addWidget(self.history_list)

    def _show_new_assessment(self):
        self.desc_input.setFocus()

    def _start_assessment(self):
        desc = self.desc_input.toPlainText().strip()
        if not desc:
            return

        ctx = self.context_input.toPlainText().strip()
        self.assess_btn.setEnabled(False)
        self.assess_btn.setText("در حال ارزیابی...")

        self._worker = RiskWorker(desc, ctx)
        self._worker.assessment_done.connect(self._on_assessment_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_assessment_done(self, result: dict):
        score = result.get("risk_score", 0)
        level = result.get("risk_level", "medium")

        level_colors = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444", "critical": "#dc2626"}
        level_names = {"low": "کم", "medium": "متوسط", "high": "بالا", "critical": "بحرانی"}

        color = level_colors.get(level, "#f59e0b")
        self.score_label.setText(str(score))
        self.score_label.setStyleSheet(f"font-size: 48px; font-weight: 700; color: {color}; background: transparent;")
        self.level_label.setText(level_names.get(level, level))
        self.score_bar.setValue(int(score))
        self.score_bar.setStyleSheet(f"QProgressBar {{ background-color: #1e293b; border-radius: 4px; }} QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}")

        html = "<div style='direction: rtl;'>"

        factors = result.get("factors", [])
        if factors:
            html += "<div style='margin-bottom: 12px;'><b style='color: #60a5fa;'>عوامل ریسک:</b><ul>"
            for f in factors:
                if isinstance(f, dict):
                    impact = f.get("impact", "medium")
                    imp_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(impact, "#f59e0b")
                    html += f"<li style='color: #e2e8f0;'><span style='color: {imp_color}; font-weight: 600;'>[{impact}]</span> {f.get('factor', '')} - {f.get('description', '')}</li>"
                else:
                    html += f"<li style='color: #e2e8f0;'>{f}</li>"
            html += "</ul></div>"

        recs = result.get("recommendations", [])
        if recs:
            html += "<div><b style='color: #4ade80;'>توصیه‌ها:</b><ul>"
            for r in recs:
                html += f"<li style='color: #e2e8f0; margin: 4px 0;'>{r}</li>"
            html += "</ul></div>"

        if result.get("summary"):
            html += f"<div style='margin-top: 12px; padding: 12px; background-color: #1e2a45; border-radius: 8px;'>{result['summary']}</div>"

        html += "</div>"
        self.results_content.setHtml(html)

        # Save to DB
        self.db.add_risk_assessment(
            title=self.desc_input.toPlainText()[:100],
            description=self.desc_input.toPlainText(),
            risk_level=level,
            risk_score=score,
            factors=result.get("factors", []),
            recommendations=result.get("recommendations", [])
        )
        self._load_assessments()

        self.assess_btn.setEnabled(True)
        self.assess_btn.setText("شروع ارزیابی ریسک")

    def _on_error(self, error: str):
        self.results_content.setPlainText(f"خطا: {error}")
        self.assess_btn.setEnabled(True)
        self.assess_btn.setText("شروع ارزیابی ریسک")

    def _load_assessments(self):
        self.history_list.clear()
        assessments = self.db.get_risk_assessments()
        level_icons = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        for a in assessments:
            icon = level_icons.get(a.get("risk_level", "medium"), "🟡")
            text = f"{icon}  {a['title'][:60]}  |  امتیاز: {a.get('risk_score', 0)}"
            item = QListWidgetItem(text)
            self.history_list.addItem(item)
