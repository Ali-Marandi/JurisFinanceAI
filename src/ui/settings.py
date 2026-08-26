from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QLineEdit, QComboBox, QSpinBox, QFormLayout,
    QGroupBox, QScrollArea, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.config import get_config


class SettingsPage(QWidget):
    """Settings page for application configuration."""

    theme_changed = pyqtSignal()

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.config = get_config()
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)

        # API Settings
        api_group = QGroupBox("تنظیمات API هوش مصنوعی")
        api_group.setStyleSheet("""
            QGroupBox {
                background-color: #1e2a45; border: 1px solid #334155; border-radius: 16px;
                margin-top: 24px; padding: 24px; padding-top: 40px; color: #f1f5f9;
                font-size: 16px; font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 20px; padding: 0 8px; color: #f8fafc;
            }
        """)
        api_layout = QFormLayout(api_group)
        api_layout.setSpacing(16)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        self.api_key_input.setStyleSheet("""
            QLineEdit {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px;
            }
            QLineEdit:focus { border-color: #3b82f6; }
        """)
        api_layout.addRow("کلید API:", self.api_key_input)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("https://api.openai.com/v1")
        self.base_url_input.setStyleSheet("""
            QLineEdit {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px;
            }
            QLineEdit:focus { border-color: #3b82f6; }
        """)
        api_layout.addRow("آدرس Base URL:", self.base_url_input)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"])
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px; min-height: 20px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #94a3b8; margin-right: 8px; }
            QComboBox QAbstractItemView { background-color: #1e2a45; color: #f1f5f9; border: 1px solid #334155; selection-background-color: #3b82f6; }
        """)
        api_layout.addRow("مدل:", self.model_combo)

        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(0, 100)
        self.temp_spin.setValue(70)
        self.temp_spin.setSuffix("%")
        self.temp_spin.setStyleSheet("""
            QSpinBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px;
            }
        """)
        api_layout.addRow("دما (خلاقیت):", self.temp_spin)

        layout.addWidget(api_group)

        # UI Settings
        ui_group = QGroupBox("تنظیمات رابط کاربری")
        ui_group.setStyleSheet("""
            QGroupBox {
                background-color: #1e2a45; border: 1px solid #334155; border-radius: 16px;
                margin-top: 24px; padding: 24px; padding-top: 40px; color: #f1f5f9;
                font-size: 16px; font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 20px; padding: 0 8px; color: #f8fafc;
            }
        """)
        ui_layout = QFormLayout(ui_group)
        ui_layout.setSpacing(16)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["تاریک", "روشن"])
        self.theme_combo.setStyleSheet("""
            QComboBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px; min-height: 20px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #94a3b8; margin-right: 8px; }
            QComboBox QAbstractItemView { background-color: #1e2a45; color: #f1f5f9; border: 1px solid #334155; selection-background-color: #3b82f6; }
        """)
        ui_layout.addRow("تم:", self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 24)
        self.font_size_spin.setValue(12)
        self.font_size_spin.setStyleSheet("""
            QSpinBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px;
            }
        """)
        ui_layout.addRow("اندازه فونت:", self.font_size_spin)

        layout.addWidget(ui_group)

        # General Settings
        general_group = QGroupBox("تنظیمات عمومی")
        general_group.setStyleSheet("""
            QGroupBox {
                background-color: #1e2a45; border: 1px solid #334155; border-radius: 16px;
                margin-top: 24px; padding: 24px; padding-top: 40px; color: #f1f5f9;
                font-size: 16px; font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 20px; padding: 0 8px; color: #f8fafc;
            }
        """)
        general_layout = QFormLayout(general_group)
        general_layout.setSpacing(16)

        self.history_spin = QSpinBox()
        self.history_spin.setRange(10, 1000)
        self.history_spin.setValue(100)
        self.history_spin.setSingleStep(10)
        self.history_spin.setStyleSheet("""
            QSpinBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px;
            }
        """)
        general_layout.addRow("حداکثر تاریخچه:", self.history_spin)

        self.export_combo = QComboBox()
        self.export_combo.addItems(["PDF", "DOCX", "TXT"])
        self.export_combo.setStyleSheet("""
            QComboBox {
                background-color: #0d1b36; color: #f1f5f9; border: 1px solid #334155;
                border-radius: 10px; padding: 10px 16px; font-size: 13px; min-height: 20px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #94a3b8; margin-right: 8px; }
            QComboBox QAbstractItemView { background-color: #1e2a45; color: #f1f5f9; border: 1px solid #334155; selection-background-color: #3b82f6; }
        """)
        general_layout.addRow("فرمت خروجی:", self.export_combo)

        layout.addWidget(general_group)

        # Save button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("ذخیره تنظیمات")
        save_btn.setFixedSize(200, 44)
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton { background-color: #3b82f6; color: white; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; }
            QPushButton:hover { background-color: #2563eb; }
        """)
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("بازنشانی")
        reset_btn.setFixedSize(120, 44)
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #94a3b8; border: 1px solid #334155; border-radius: 10px; font-size: 14px; }
            QPushButton:hover { border-color: #ef4444; color: #f87171; }
        """)
        reset_btn.clicked.connect(self._reset_settings)
        btn_layout.addWidget(reset_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _load_settings(self):
        self.api_key_input.setText(self.config.get("api.openai_api_key", ""))
        self.base_url_input.setText(self.config.get("api.openai_base_url", "https://api.openai.com/v1"))

        model = self.config.get("api.model", "gpt-4o-mini")
        idx = self.model_combo.findText(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        temp = int(self.config.get("api.temperature", 0.7) * 100)
        self.temp_spin.setValue(temp)

        theme = self.config.get("ui.theme", "dark")
        self.theme_combo.setCurrentIndex(0 if theme == "dark" else 1)

        self.font_size_spin.setValue(self.config.get("ui.font_size", 12))
        self.history_spin.setValue(self.config.get("general.max_history", 100))

        export_fmt = self.config.get("general.export_format", "pdf").upper()
        idx = self.export_combo.findText(export_fmt)
        if idx >= 0:
            self.export_combo.setCurrentIndex(idx)

    def _save_settings(self):
        self.config.set("api.openai_api_key", self.api_key_input.text())
        self.config.set("api.openai_base_url", self.base_url_input.text())
        self.config.set("api.model", self.model_combo.currentText())
        self.config.set("api.temperature", self.temp_spin.value() / 100.0)

        theme_value = "dark" if self.theme_combo.currentIndex() == 0 else "light"
        self.config.set("ui.theme", theme_value)
        self.config.set("ui.font_size", self.font_size_spin.value())
        self.config.set("general.max_history", self.history_spin.value())
        self.config.set("general.export_format", self.export_combo.currentText().lower())

        QMessageBox.information(self, "ذخیره شد", "تنظیمات با موفقیت ذخیره شد.")

        if self.main_window:
            self.main_window.refresh_theme()

    def _reset_settings(self):
        reply = QMessageBox.question(
            self, "بازنشانی",
            "آیا مطمئن هستید؟ تمام تنظیمات به حالت پیش‌فرض برمی‌گردد.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config.config = self.config.DEFAULT_CONFIG.copy()
            self.config.save()
            self._load_settings()
