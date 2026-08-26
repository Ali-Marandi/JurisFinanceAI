import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QTextEdit, QFileDialog, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.database import get_database
from ..core.document_parser import DocumentParser
from ..core.ai_engine import get_ai_engine


class ContractWorker(QThread):
    """Worker thread for contract analysis."""
    analysis_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def run(self):
        try:
            ai = get_ai_engine()
            result = ai.analyze_contract(self.text)
            self.analysis_done.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ContractsPage(QWidget):
    """Contract analysis page with AI-powered review."""

    def __init__(self):
        super().__init__()
        self.db = get_database()
        self._contract_text = ""
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Header
        header = QFrame()
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(4)
        title = QLabel("تحلیل هوشمند قراردادها")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f8fafc; background: transparent;")
        header_layout.addWidget(title)
        subtitle = QLabel("قرارداد خود را آپلود یا وارد کنید تا هوش مصنوعی آن را تحلیل کند")
        subtitle.setStyleSheet("font-size: 13px; color: #64748b; background: transparent;")
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header)

        # Upload and actions bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(12)

        upload_btn = QPushButton("  آپلود فایل قرارداد")
        upload_btn.setFixedHeight(42)
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.setStyleSheet("""
            QPushButton { background-color: #3b82f6; color: white; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; }
            QPushButton:hover { background-color: #2563eb; }
        """)
        upload_btn.clicked.connect(self._upload_contract)
        action_bar.addWidget(upload_btn)

        self.analyze_btn = QPushButton("  شروع تحلیل")
        self.analyze_btn.setFixedHeight(42)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.setStyleSheet("""
            QPushButton { background-color: #22c55e; color: white; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; }
            QPushButton:hover { background-color: #16a34a; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
        """)
        self.analyze_btn.clicked.connect(self._analyze)
        action_bar.addWidget(self.analyze_btn)

        action_bar.addStretch()
        main_layout.addLayout(action_bar)

        # Content area - two columns
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Input column
        input_card = QFrame()
        input_card.setObjectName("card")
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(16, 16, 16, 16)

        input_title = QLabel("متن قرارداد")
        input_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #f8fafc; background: transparent;")
        input_layout.addWidget(input_title)

        self.contract_input = QTextEdit()
        self.contract_input.setPlaceholderText("متن قرارداد را اینجا وارد کنید یا فایل آپلود کنید...\n\nمثال:\nقرارداد اجاره به شماره ۱۲۳۴ مورخ ۱۴۰۳/۰۶/۱۵\nبین موجر: علی محمدی و مستاجر: رضا احمدی\n...")
        self.contract_input.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 16px; font-size: 13px; line-height: 1.7; }
            QTextEdit:focus { border-color: #3b82f6; }
        """)
        self.contract_input.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self.contract_input)

        content_layout.addWidget(input_card, 1)

        # Results column
        results_card = QFrame()
        results_card.setObjectName("card")
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(16, 16, 16, 16)

        results_title = QLabel("نتایج تحلیل")
        results_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #f8fafc; background: transparent;")
        results_layout.addWidget(results_title)

        self.results_content = QTextEdit()
        self.results_content.setReadOnly(True)
        self.results_content.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; font-size: 13px; }
        """)
        self.results_content.setHtml("""
            <div style='text-align: center; padding: 60px 20px;'>
                <div style='font-size: 48px; margin-bottom: 16px;'>🛡</div>
                <div style='font-size: 16px; color: #64748b;'>نتایج تحلیل اینجا نمایش داده می‌شود</div>
                <div style='font-size: 13px; color: #475569; margin-top: 8px;'>یک قرارداد وارد و تحلیل را شروع کنید</div>
            </div>
        """)
        results_layout.addWidget(self.results_content)

        content_layout.addWidget(results_card, 1)

        main_layout.addLayout(content_layout, 1)

    def _on_text_changed(self):
        text = self.contract_input.toPlainText().strip()
        self.analyze_btn.setEnabled(len(text) > 50)
        self._contract_text = text

    def _upload_contract(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "انتخاب فایل قرارداد", "",
            "Documents (*.pdf *.docx *.doc *.txt);;All Files (*)"
        )
        if not file_path:
            return

        text, metadata = DocumentParser.parse(file_path)
        if text:
            self.contract_input.setPlainText(text[:15000])
        else:
            QMessageBox.warning(self, "خطا", "خطا در خواندن فایل")

    def _analyze(self):
        if not self._contract_text:
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("در حال تحلیل...")
        self.results_content.setHtml("<div style='text-align: center; padding: 40px; color: #64748b;'>در حال تحلیل قرارداد با هوش مصنوعی...</div>")

        self._worker = ContractWorker(self._contract_text[:8000])
        self._worker.analysis_done.connect(self._on_analysis_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_analysis_done(self, result: dict):
        self._format_results(result)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("  شروع تحلیل")

    def _on_error(self, error: str):
        self.results_content.setPlainText(f"خطا: {error}")
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("  شروع تحلیل")

    def _format_results(self, result: dict):
        html = "<div style='font-family: Segoe UI; direction: rtl;'>"

        # Score
        score = result.get("overall_score", 0)
        score_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
        html += f"""
            <div style='text-align: center; margin-bottom: 20px; padding: 16px; background-color: #1e2a45; border-radius: 12px;'>
                <div style='font-size: 13px; color: #94a3b8;'>امتیاز کلی قرارداد</div>
                <div style='font-size: 36px; font-weight: bold; color: {score_color};'>{score}/100</div>
            </div>
        """

        # Contract type
        if result.get("contract_type"):
            html += f"<div style='margin-bottom: 12px;'><b style='color: #60a5fa;'>نوع قرارداد:</b> {result['contract_type']}</div>"

        # Parties
        parties = result.get("parties", [])
        if parties:
            html += "<div style='margin-bottom: 12px;'><b style='color: #60a5fa;'>طرفین:</b> " + ", ".join(parties) + "</div>"

        # Key clauses
        clauses = result.get("key_clauses", [])
        if clauses:
            html += "<div style='margin-bottom: 12px;'><b style='color: #60a5fa;'>بندهای کلیدی:</b><ol>"
            for c in clauses:
                html += f"<li style='color: #e2e8f0; margin: 4px 0;'>{c}</li>"
            html += "</ol></div>"

        # Risks
        risks = result.get("risks", [])
        if risks:
            html += "<div style='margin-bottom: 12px;'><b style='color: #f87171;'>ریسک‌ها:</b><ul>"
            for r in risks:
                if isinstance(r, dict):
                    level = r.get("level", "medium")
                    desc = r.get("description", "")
                    color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(level, "#f59e0b")
                    html += f"<li style='color: #e2e8f0;'><span style='color: {color}; font-weight: 600;'>[{level.upper()}]</span> {desc}</li>"
                else:
                    html += f"<li style='color: #e2e8f0;'>{r}</li>"
            html += "</ul></div>"

        # Recommendations
        recs = result.get("recommendations", [])
        if recs:
            html += "<div style='margin-bottom: 12px;'><b style='color: #4ade80;'>پیشنهادات:</b><ul>"
            for r in recs:
                html += f"<li style='color: #e2e8f0; margin: 4px 0;'>{r}</li>"
            html += "</ul></div>"

        # Summary
        if result.get("summary"):
            html += f"<div style='margin-top: 16px; padding: 12px; background-color: #1e2a45; border-radius: 10px;'><b style='color: #60a5fa;'>خلاصه:</b> {result['summary']}</div>"

        html += "</div>"
        self.results_content.setHtml(html)
