"""
JurisFinanceAI - AI Chat Page
Chat interface for legal and financial AI assistance.
"""

import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QTextEdit, QComboBox, QScrollArea, QListWidget,
    QListWidgetItem, QSplitter, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QTextCursor

from ..core.database import get_database
from ..core.ai_engine import get_ai_engine


class ChatWorker(QThread):
    """Worker thread for AI responses."""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, message: str, history: list, category: str):
        super().__init__()
        self.message = message
        self.history = history
        self.category = category

    def run(self):
        try:
            ai = get_ai_engine()
            response = ai.chat(self.message, self.history, self.category)
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatBubble(QFrame):
    """A chat message bubble."""
    def __init__(self, text: str, is_user: bool = True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setObjectName("chatBubbleUser" if is_user else "chatBubbleAI")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        role_label = QLabel("شما" if is_user else "JurisFinanceAI")
        role_label.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {'#60a5fa' if not is_user else '#94a3b8'}; background: transparent;")
        layout.addWidget(role_label)

        content = QLabel(text)
        content.setWordWrap(True)
        content.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content.setStyleSheet(f"font-size: 13px; line-height: 1.6; color: {'#e2e8f0' if not is_user else '#f1f5f9'}; background: transparent;")
        layout.addWidget(content)

        if is_user:
            self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)


class AIChatPage(QWidget):
    """AI Chat page with conversation management."""

    def __init__(self):
        super().__init__()
        self.db = get_database()
        self.ai = get_ai_engine()
        self.current_conversation_id = None
        self.conversation_history = []
        self._is_generating = False
        self._init_ui()
        self._load_conversations()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel - conversation list
        left_panel = QFrame()
        left_panel.setFixedWidth(280)
        left_panel.setStyleSheet("background-color: #16213e; border-left: 1px solid #1e293b;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        new_btn = QPushButton("+  گفتگوی جدید")
        new_btn.setFixedHeight(40)
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white; border: none; border-radius: 10px;
                font-size: 13px; font-weight: 600; text-align: center;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        new_btn.clicked.connect(self._new_conversation)
        left_layout.addWidget(new_btn)

        self.category_combo = QComboBox()
        self.category_combo.addItems(["همه", "حقوقی", "مالی", "قراردادها", "عمومی"])
        self.category_combo.setFixedHeight(36)
        self.category_combo.setStyleSheet("""
            QComboBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155; border-radius: 8px;
                padding: 4px 12px; font-size: 12px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #94a3b8; margin-right: 8px; }
            QComboBox QAbstractItemView { background-color: #1e2a45; color: #f1f5f9; border: 1px solid #334155; selection-background-color: #3b82f6; }
        """)
        left_layout.addWidget(self.category_combo)

        self.conv_list = QListWidget()
        self.conv_list.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; color: #f1f5f9; outline: none; }
            QListWidget::item { padding: 8px 10px; border-radius: 8px; margin: 2px 0; }
            QListWidget::item:hover { background-color: #1a2744; }
            QListWidget::item:selected { background-color: #1a3a6e; }
        """)
        self.conv_list.currentRowChanged.connect(self._on_conversation_selected)
        left_layout.addWidget(self.conv_list)

        main_layout.addWidget(left_panel)

        # Right panel - chat area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Chat type selector bar
        type_bar = QFrame()
        type_bar.setFixedHeight(48)
        type_bar.setStyleSheet("background-color: #16213e; border-bottom: 1px solid #1e293b;")
        type_layout = QHBoxLayout(type_bar)
        type_layout.setContentsMargins(20, 0, 20, 0)

        self.chat_type_buttons = []
        types = [("legal", "حقوقی"), ("financial", "مالی"), ("contract", "قرارداد"), ("general", "عمومی")]
        for i, (key, label) in enumerate(types):
            btn = QPushButton(label)
            btn.setFixedHeight(36)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("type_key", key)
            is_first = i == 0
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {"#3b82f6" if is_first else "transparent"};
                    color: {"#ffffff" if is_first else "#94a3b8"};
                    border: none; border-radius: 8px; padding: 0 16px; font-size: 12px; font-weight: 500;
                }}
                QPushButton:hover {{ background-color: {"#2563eb" if is_first else "#1a2744"}; }}
            """)
            btn.clicked.connect(lambda checked, k=key: self._set_chat_type(k))
            type_layout.addWidget(btn)
            self.chat_type_buttons.append(btn)

        self.current_chat_type = "legal"
        type_layout.addStretch()
        right_layout.addWidget(type_bar)

        # Messages scroll area
        self.messages_scroll = QScrollArea()
        self.messages_scroll.setWidgetResizable(True)
        self.messages_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(24, 20, 24, 20)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch()

        # Welcome message
        self.welcome_widget = self._create_welcome_widget()
        self.messages_layout.insertWidget(0, self.welcome_widget)

        self.messages_scroll.setWidget(self.messages_container)
        right_layout.addWidget(self.messages_scroll)

        # Input area
        input_area = QFrame()
        input_area.setFixedHeight(80)
        input_area.setStyleSheet("background-color: #16213e; border-top: 1px solid #1e293b;")
        input_layout = QHBoxLayout(input_area)
        input_layout.setContentsMargins(20, 12, 20, 12)
        input_layout.setSpacing(12)

        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("سوال خود را بنویسید...")
        self.input_field.setMaximumHeight(56)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 12px; padding: 12px 16px; font-size: 13px;
            }
            QTextEdit:focus { border-color: #3b82f6; }
        """)
        self.input_field.installEventFilter(self)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("ارسال")
        self.send_btn.setFixedSize(100, 56)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white; border: none; border-radius: 12px;
                font-size: 13px; font-weight: 600;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
        """)
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)

        right_layout.addWidget(input_area)
        main_layout.addWidget(right_panel, 1)

    def _create_welcome_widget(self):
        welcome = QFrame()
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.setSpacing(12)

        icon = QLabel("⚖")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 48px; background: transparent;")
        welcome_layout.addWidget(icon)

        title = QLabel("JurisFinanceAI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #f8fafc; background: transparent;")
        welcome_layout.addWidget(title)

        subtitle = QLabel("دستیار هوش مصنوعی حقوقی و مالی شما")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #64748b; background: transparent;")
        welcome_layout.addWidget(subtitle)

        prompts_layout = QHBoxLayout()
        prompts_layout.setSpacing(8)
        prompts = ["شرح ماده ۱۰ قانون مدنی", "تحلیل ریسک سرمایه‌گذاری", "نکات کلیدی قرارداد اجاره", "حقوق مستأجر چیست؟"]
        for p in prompts:
            btn = QPushButton(p)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e2a45; color: #94a3b8; border: 1px solid #334155;
                    border-radius: 10px; padding: 8px 14px; font-size: 12px;
                }
                QPushButton:hover { border-color: #3b82f6; color: #f1f5f9; }
            """)
            btn.clicked.connect(lambda checked, text=p: self._send_quick_prompt(text))
            prompts_layout.addWidget(btn)
        welcome_layout.addLayout(prompts_layout)

        return welcome

    def eventFilter(self, obj, event):
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _new_conversation(self):
        conv_id = self.db.create_conversation(category=self.current_chat_type)
        self.current_conversation_id = conv_id
        self.conversation_history = []
        self._load_conversations()
        self._clear_messages()
        self._show_welcome()

    def _load_conversations(self):
        self.conv_list.clear()
        convs = self.db.get_conversations()
        for c in convs:
            self.conv_list.addItem(c["title"])

    def _on_conversation_selected(self, row):
        if row < 0:
            return
        convs = self.db.get_conversations()
        if row < len(convs):
            conv = convs[row]
            self.current_conversation_id = conv["id"]
            self._load_messages(conv["id"])

    def _load_messages(self, conv_id):
        self._clear_messages()
        messages = self.db.get_conversation_messages(conv_id)
        self.conversation_history = []
        for msg in messages:
            bubble = ChatBubble(msg["content"], msg["role"] == "user")
            self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
            if msg["role"] != "system":
                self.conversation_history.append({"role": msg["role"], "content": msg["content"]})
        self._scroll_to_bottom()

    def _clear_messages(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_welcome(self):
        self._clear_messages()
        self.messages_layout.insertWidget(0, self.welcome_widget)
        self.welcome_widget.show()

    def _set_chat_type(self, chat_type: str):
        self.current_chat_type = chat_type
        for btn in self.chat_type_buttons:
            is_active = btn.property("type_key") == chat_type
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {"#3b82f6" if is_active else "transparent"};
                    color: {"#ffffff" if is_active else "#94a3b8"};
                    border: none; border-radius: 8px; padding: 0 16px; font-size: 12px; font-weight: 500;
                }}
                QPushButton:hover {{ background-color: {"#2563eb" if is_active else "#1a2744"}; }}
            """)

    def _send_quick_prompt(self, text: str):
        self.input_field.setPlainText(text)
        self._send_message()

    def _send_message(self):
        if self._is_generating:
            return

        text = self.input_field.toPlainText().strip()
        if not text:
            return

        if self.welcome_widget.isVisible():
            self.welcome_widget.hide()

        if self.current_conversation_id is None:
            self.current_conversation_id = self.db.create_conversation(
                title=text[:50] + ("..." if len(text) > 50 else ""),
                category=self.current_chat_type
            )
            self._load_conversations()

        self.db.add_message(self.current_conversation_id, "user", text)
        self.conversation_history.append({"role": "user", "content": text})

        user_bubble = ChatBubble(text, is_user=True)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, user_bubble)

        self.input_field.clear()
        self._scroll_to_bottom()

        loading = QLabel("در حال تحلیل...")
        loading.setStyleSheet("font-size: 13px; color: #64748b; background: transparent; padding: 10px;")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, loading)
        self._scroll_to_bottom()

        self._is_generating = True
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)

        self._worker = ChatWorker(text, self.conversation_history.copy(), self.current_chat_type)
        self._worker.response_ready.connect(lambda r: self._on_response_ready(r, loading))
        self._worker.error_occurred.connect(lambda e: self._on_error(e, loading))
        self._worker.start()

    def _on_response_ready(self, response: str, loading_label: QLabel):
        loading_label.deleteLater()
        self.db.add_message(self.current_conversation_id, "assistant", response)
        self.conversation_history.append({"role": "assistant", "content": response})
        ai_bubble = ChatBubble(response, is_user=False)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, ai_bubble)
        self._is_generating = False
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self._scroll_to_bottom()

    def _on_error(self, error: str, loading_label: QLabel):
        loading_label.deleteLater()
        error_bubble = ChatBubble(f"خطا: {error}", is_user=False)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, error_bubble)
        self._is_generating = False
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)

    def _scroll_to_bottom(self):
        QTimer.singleShot(100, lambda: self.messages_scroll.verticalScrollBar().setValue(
            self.messages_scroll.verticalScrollBar().maximum()
        ))
