"""Tests for ConfigManager (src/core/config.py).

Uses monkeypatching to redirect the config directory to a temp path.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config import ConfigManager


class TestConfigManager:
    """ConfigManager tests – isolated to a temp directory."""

    @pytest.fixture
    def config_mgr(self, tmp_path):
        """Create a ConfigManager that uses *tmp_path* instead of ~/.jurisfinanceai."""
        with patch.object(ConfigManager, "_ensure_dirs", lambda self: None):
            mgr = ConfigManager.__new__(ConfigManager)
            mgr.app_dir = tmp_path
            mgr.config_file = tmp_path / "config.json"
            mgr.key_file = tmp_path / ".key"
            # Create our own key so we don't touch the real one
            from cryptography.fernet import Fernet
            mgr._key = Fernet.generate_key()
            # Start with default config (no file on disk yet)
            mgr.config = ConfigManager.DEFAULT_CONFIG.copy()
            # Ensure no real file exists
            if mgr.config_file.exists():
                mgr.config_file.unlink()
            return mgr

    def test_default_config(self, config_mgr):
        """Loads with defaults when no config file exists."""
        assert config_mgr.get("api.model") == "gpt-4o-mini"
        assert config_mgr.get("ui.theme") == "dark"
        assert config_mgr.get("general.auto_save") is True

    def test_set_get(self, config_mgr):
        """set and get values round-trip correctly."""
        config_mgr.set("ui.theme", "light")
        assert config_mgr.get("ui.theme") == "light"

        config_mgr.set("general.max_history", 500)
        assert config_mgr.get("general.max_history") == 500

    def test_encryption(self, config_mgr):
        """Encrypted values survive round-trip (save → reload)."""
        secret = "sk-abc123secretkey"
        config_mgr.set("api.openai_api_key", secret)
        config_mgr.save()

        # Verify the file has an encrypted value (not the plain secret)
        raw = json.loads(config_mgr.config_file.read_text())
        assert raw["api"]["openai_api_key"] != secret
        assert len(raw["api"]["openai_api_key"]) > 0

        # Reload and verify decryption works
        with patch.object(ConfigManager, "_ensure_dirs", lambda self: None):
            mgr2 = ConfigManager.__new__(ConfigManager)
            mgr2.app_dir = config_mgr.app_dir
            mgr2.config_file = config_mgr.config_file
            mgr2.key_file = config_mgr.key_file
            mgr2._key = config_mgr._key
            mgr2.config = mgr2._load_config()

        assert mgr2.get("api.openai_api_key") == secret
