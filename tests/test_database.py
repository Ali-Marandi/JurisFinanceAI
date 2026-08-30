"""Tests for DatabaseManager (src/core/database.py).

Uses a temporary database in a temp directory to avoid touching real data.
"""

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: build a standalone DatabaseManager that writes to tmp_path
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Create a DatabaseManager pointed at a temp DB."""
    # We avoid importing DatabaseManager at module level because it
    # triggers get_config() at import time via the singleton path.
    from core.database import DatabaseManager

    db_path = tmp_path / "test.db"
    # Patch the path after construction
    with patch.object(DatabaseManager, "_create_tables", lambda self: None):
        mgr = DatabaseManager.__new__(DatabaseManager)
    mgr.conn = sqlite3.connect(str(db_path), check_same_thread=False)
    mgr.conn.row_factory = sqlite3.Row
    mgr.conn.execute("PRAGMA journal_mode=WAL")
    mgr.conn.execute("PRAGMA busy_timeout=5000")
    mgr._lock = threading.Lock()
    mgr.db_path = db_path
    # Actually create tables now
    mgr._create_tables()
    return mgr, db_path


class TestDatabase:
    """DatabaseManager tests isolated to a temp file."""

    @pytest.fixture
    def db(self, tmp_path):
        mgr, path = _make_db(tmp_path)
        yield mgr
        mgr.close()

    def test_create_tables(self, db, tmp_path):
        """Tables are created successfully."""
        cursor = db.conn.cursor()
        # Query sqlite_master to check tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        assert "conversations" in tables
        assert "messages" in tables
        assert "documents" in tables
        assert "cases" in tables
        assert "financial_records" in tables
        assert "risk_assessments" in tables
        assert "reports" in tables
        assert "analytics" in tables

    def test_add_log(self, db):
        """Logging actions inserts into analytics table."""
        db.log_action("portfolio_optimize", {"assets": 5})
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM analytics")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["action"] == "portfolio_optimize"
        details = json.loads(rows[0]["details"])
        assert details["assets"] == 5

    def test_conversations(self, db):
        """Create and retrieve conversations."""
        conv_id = db.create_conversation(title="Test Conv", category="finance")
        assert isinstance(conv_id, int)
        assert conv_id > 0

        # Add a message
        msg_id = db.add_message(conv_id, "user", "Hello", tokens_used=10, model="gpt-4")
        assert isinstance(msg_id, int)

        # Retrieve conversations
        convs = db.get_conversations()
        assert len(convs) >= 1
        found = [c for c in convs if c["id"] == conv_id]
        assert len(found) == 1
        assert found[0]["title"] == "Test Conv"

        # Get messages for this conversation
        msgs = db.get_conversation_messages(conv_id)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
