"""
JurisFinanceAI - Configuration Management
Manages application settings, API keys, and user preferences.
"""

import json
import os
from pathlib import Path
from cryptography.fernet import Fernet
import base64
import hashlib


class ConfigManager:
    """Manages all application configuration securely."""

    DEFAULT_CONFIG = {
        "api": {
            "openai_api_key": "",
            "openai_base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "ui": {
            "theme": "dark",
            "language": "fa",
            "font_size": 12,
            "sidebar_collapsed": False,
            "window_geometry": None,
        },
        "general": {
            "auto_save": True,
            "save_interval": 300,
            "max_history": 100,
            "export_format": "pdf",
        },
        "recent_files": [],
        "pinned_queries": [],
    }

    def __init__(self):
        self.app_dir = Path.home() / ".jurisfinanceai"
        self.config_file = self.app_dir / "config.json"
        self.key_file = self.app_dir / ".key"
        self._ensure_dirs()
        self._key = self._get_or_create_key()
        self.config = self._load_config()

    def _ensure_dirs(self):
        """Create application directories if they don't exist."""
        self.app_dir.mkdir(parents=True, exist_ok=True)
        (self.app_dir / "documents").mkdir(exist_ok=True)
        (self.app_dir / "reports").mkdir(exist_ok=True)
        (self.app_dir / "cache").mkdir(exist_ok=True)
        (self.app_dir / "logs").mkdir(exist_ok=True)

    def _get_or_create_key(self):
        """Get or create encryption key for sensitive data."""
        if self.key_file.exists():
            return self.key_file.read_bytes()
        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        return key

    def _encrypt(self, value: str) -> str:
        """Encrypt a string value."""
        if not value:
            return ""
        f = Fernet(self._key)
        return f.encrypt(value.encode()).decode()

    def _decrypt(self, value: str) -> str:
        """Decrypt a string value."""
        if not value:
            return ""
        try:
            f = Fernet(self._key)
            return f.decrypt(value.encode()).decode()
        except Exception:
            return value

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    config = self.DEFAULT_CONFIG.copy()
                    config.update(loaded)
                    # Decrypt sensitive fields
                    if "api" in config and "openai_api_key" in config["api"]:
                        config["api"]["openai_api_key"] = self._decrypt(
                            config["api"]["openai_api_key"]
                        )
                    return config
            except (json.JSONDecodeError, KeyError):
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()

    def save(self):
        """Save current configuration to file."""
        save_data = json.loads(json.dumps(self.config))
        # Encrypt sensitive fields before saving
        if "api" in save_data and "openai_api_key" in save_data["api"]:
            save_data["api"]["openai_api_key"] = self._encrypt(
                save_data["api"]["openai_api_key"]
            )
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

    def get(self, key: str, default=None):
        """Get a config value by dot-notation key."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value):
        """Set a config value by dot-notation key."""
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save()

    def add_recent_file(self, filepath: str):
        """Add a file to recent files list."""
        recent = self.config.get("recent_files", [])
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        self.config["recent_files"] = recent[:20]
        self.save()

    @property
    def app_directory(self) -> Path:
        return self.app_dir

    @property
    def documents_dir(self) -> Path:
        return self.app_dir / "documents"

    @property
    def reports_dir(self) -> Path:
        return self.app_dir / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.app_dir / "cache"


# Singleton instance
_config_instance = None


def get_config() -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
