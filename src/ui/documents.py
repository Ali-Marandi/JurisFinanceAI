"""
JurisFinanceAI - Documents Management Page
Upload, manage, and analyze legal/financial documents.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QProgressBar, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from ..core.database import get_database
from ..core.document_parser import DocumentParser
from ..core.ai_engine import get_ai_engine


class DocumentWorker(QThread):
    """Worker thread for document analysis."""
    analysis_done = pyqtSignal(int, str, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, doc_id: int, text: str, analysis_type: str):
        super().__init__()
        self.doc_id = doc_id
        self.text = text
        self.analysis_type = analysis_type

    def run(self):
        try:
            ai = get_ai_engine()
            summary = ai.analyze_document(self.text, self.analysis_type)
            self.analysis_done.emit(self.doc_id, summary, {})
        except Exception as e:
            self.error_occurred.emit(str(e))


class DocumentsPage(QWidget):
    """Documents management and analysis page."""

    def __init__(self):
        super().__init__()
        self.db = get_database()
        self._init_ui()
        self._refresh_list()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Top action bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(12)

        upload_btn = QPushButton("  آپلود سند جدید")
        upload_btn.setFixedHeight(42)
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.setStyleSheet("""
            QPushButton { background-color: #3b82f6; color: white; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; }
            QPushButton:hover { background-color: #2563eb; }
        """)
        upload_btn.clicked.connect(self._upload_document)
        action_bar.addWidget(upload_btn)

        refresh_btn = QPushButton("  بروزرسانی")
        refresh_btn.setFixedHeight(42)
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #94a3b8; border: 1px solid #334155; border-radius: 10px; font-size: 13px; }
            QPushButton:hover { border-color: #3b82f6; color: #f1f5f9; }
        """)
        refresh_btn.clicked.connect(self._refresh_list)
        action_bar.addWidget(refresh_btn)

        action_bar.addStretch()

        # Search
        self.search_input = QTextEdit()
        self.search_input.setPlaceholderText("جستجو در اسناد...")
        self.search_input.setFixedSize(300, 42)
        self.search_input.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155; border-radius: 10px; padding: 8px 14px; font-size: 13px; }
            QTextEdit:focus { border-color: #3b82f6; }
        """)
        self.search_input.textChanged.connect(self._on_search_changed)
        action_bar.addWidget(self.search_input)

        main_layout.addLayout(action_bar)

        # Splitter: list + detail
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Document list
        list_card = QFrame()
        list_card.setObjectName("card")
        list_layout = QVBoxLayout(list_card)
        list_layout.setContentsMargins(16, 16, 16, 16)

        list_title = QLabel("اسناد")
        list_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent;")
        list_layout.addWidget(list_title)

        self.doc_list = QListWidget()
        self.doc_list.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; color: #f1f5f9; outline: none; }
            QListWidget::item { padding: 12px; border-radius: 8px; margin: 2px 0; }
            QListWidget::item:hover { background-color: #243352; }
            QListWidget::item:selected { background-color: #1a3a6e; }
        """)
        self.doc_list.currentRowChanged.connect(self._on_doc_selected)
        list_layout.addWidget(self.doc_list)

        splitter.addWidget(list_card)

        # Document detail panel
        detail_card = QFrame()
        detail_card.setObjectName("card")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(16, 16, 16, 16)

        self.detail_title = QLabel("یک سند را انتخاب کنید")
        self.detail_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #f8fafc; background: transparent;")
        detail_layout.addWidget(self.detail_title)

        self.detail_content = QTextEdit()
        self.detail_content.setReadOnly(True)
        self.detail_content.setStyleSheet("""
            QTextEdit { background-color: #0d1b36; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; font-size: 13px; line-height: 1.7; }
        """)
        detail_layout.addWidget(self.detail_content)

        # Analysis action bar
        analysis_bar = QHBoxLayout()
        analysis_bar.setSpacing(8)

        self.analyze_btn = QPushButton("تحلیل با هوش مصنوعی")
        self.analyze_btn.setFixedHeight(40)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.setStyleSheet("""
            QPushButton { background-color: #8b5cf6; color: white; border: none; border-radius: 10px; font-size: 13px; font-weight: 500; }
            QPushButton:hover { background-color: #7c3aed; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
        """)
        self.analyze_btn.clicked.connect(self._analyze_document)
        analysis_bar.addWidget(self.analyze_btn)

        self.delete_btn = QPushButton("حذف")
        self.delete_btn.setFixedSize(80, 40)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #f87171; border: 1px solid #7f1d1d40; border-radius: 10px; font-size: 13px; }
            QPushButton:hover { background-color: #7f1d1d20; border-color: #ef4444; }
            QPushButton:disabled { color: #475569; border-color: #334155; }
        """)
        self.delete_btn.clicked.connect(self._delete_document)
        analysis_bar.addWidget(self.delete_btn)

        analysis_bar.addStretch()
        detail_layout.addLayout(analysis_bar)

        splitter.addWidget(detail_card)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter, 1)

        self.current_doc_id = None
        self.current_doc_text = None

    def _upload_document(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "انتخاب سند", "",
            "Documents (*.pdf *.docx *.doc *.txt *.rtf);;All Files (*)"
        )
        if not file_path:
            return

        if not DocumentParser.is_supported(file_path):
            QMessageBox.warning(self, "خطا", "فرمت فایل پشتیبانی نمی‌شود.\nفرمت‌های پشتیبانی: PDF, DOCX, TXT, RTF")
            return

        # Parse document
        text, metadata = DocumentParser.parse(file_path)
        if text is None:
            QMessageBox.warning(self, "خطا", f"خطا در خواندن فایل: {metadata.get('error', 'نامشخص')}")
            return

        filename = os.path.basename(file_path)
        file_type = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(file_path)

        doc_id = self.db.add_document(filename, file_path, file_type, file_size, text)
        self.db.update_document_analysis(doc_id, text[:500], metadata)
        try:
            from ..core.config import get_config
            self.config = get_config()
            self.config.add_recent_file(file_path)
        except Exception:
            pass
        self.db.log_action("upload_document", {"filename": filename})

        self._refresh_list()

    def _refresh_list(self):
        self.doc_list.clear()
        docs = self.db.get_documents()
        for doc in docs:
            icon_text = {".pdf": "📕", ".docx": "📘", ".doc": "📘", ".txt": "📄"}.get(doc["file_type"], "📄")
            item = QListWidgetItem(f"{icon_text}  {doc['filename']}  ({doc['file_type'].upper()})")
            item.setData(Qt.ItemDataRole.UserRole, doc["id"])
            self.doc_list.addItem(item)

    def _on_doc_selected(self, row):
        if row < 0:
            self.analyze_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.detail_title.setText("یک سند را انتخاب کنید")
            self.detail_content.clear()
            self.current_doc_id = None
            return

        item = self.doc_list.item(row)
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        docs = self.db.get_documents()
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if not doc:
            return

        self.current_doc_id = doc_id
        self.current_doc_text = doc.get("content", "")
        self.detail_title.setText(doc["filename"])
        self.detail_content.setPlainText(doc.get("summary") or doc.get("content", "محتوایی یافت نشد.")[:5000])
        self.analyze_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)

    def _analyze_document(self):
        if not self.current_doc_id or not self.current_doc_text:
            return

        # Clean up previous worker
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(1000)

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("در حال تحلیل...")

        self._worker = DocumentWorker(self.current_doc_id, self.current_doc_text, "legal")
        self._worker.analysis_done.connect(self._on_analysis_done)
        self._worker.error_occurred.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_done(self, doc_id, summary, metadata):
        self.db.update_document_analysis(doc_id, summary, metadata)
        self.detail_content.setPlainText(summary)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("تحلیل با هوش مصنوعی")

    def _on_analysis_error(self, error):
        self.detail_content.setPlainText(f"خطا در تحلیل: {error}")
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("تحلیل با هوش مصنوعی")

    def _on_search_changed(self, text: str):
        """Filter document list based on search text."""
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _delete_document(self):
        if self.current_doc_id:
            self.db.delete_document(self.current_doc_id)
            self.current_doc_id = None
            self.current_doc_text = None
            self.detail_title.setText("یک سند را انتخاب کنید")
            self.detail_content.clear()
            self.analyze_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self._refresh_list()
