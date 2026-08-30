"""
JurisFinanceAI - SQLite Database Manager
Handles all local data persistence: chat history, documents, cases, reports.
"""

import sqlite3
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from .config import get_config


class DatabaseManager:
    """Manages all database operations for JurisFinanceAI."""

    def __init__(self):
        config = get_config()
        self.db_path = config.app_directory / "jurisfinanceai.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        """Create all database tables."""
        cursor = self.conn.cursor()

        # Chat conversations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'گفتگوی جدید',
                category TEXT DEFAULT 'legal',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_pinned INTEGER DEFAULT 0
            )
        """)

        # Chat messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                model TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                content TEXT,
                summary TEXT,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                analyzed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Legal cases
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_number TEXT UNIQUE,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'closed', 'pending', 'archived')),
                priority TEXT DEFAULT 'medium' CHECK(priority IN ('low', 'medium', 'high', 'critical')),
                category TEXT DEFAULT 'civil',
                parties TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Financial records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                record_type TEXT NOT NULL CHECK(record_type IN ('income', 'expense', 'settlement', 'fee')),
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'IRR',
                description TEXT,
                date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE SET NULL
            )
        """)

        # Risk assessments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                risk_level TEXT DEFAULT 'medium' CHECK(risk_level IN ('low', 'medium', 'high', 'critical')),
                risk_score REAL DEFAULT 0.0,
                factors TEXT DEFAULT '[]',
                recommendations TEXT DEFAULT '[]',
                case_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE SET NULL
            )
        """)

        # Reports
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                report_type TEXT NOT NULL,
                content TEXT,
                file_path TEXT,
                case_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE SET NULL
            )
        """)

        # Contract analyses
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contract_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                contract_type TEXT,
                parties TEXT DEFAULT '[]',
                key_clauses TEXT DEFAULT '[]',
                risks TEXT DEFAULT '[]',
                recommendations TEXT DEFAULT '[]',
                overall_score REAL DEFAULT 0.0,
                analysis_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
            )
        """)

        # App usage analytics (local only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                details TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(file_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_case ON financial_records(case_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_case ON risk_assessments(case_id)")

        self.conn.commit()

    # ==================== Conversation Methods ====================

    def create_conversation(self, title: str = "گفتگوی جدید", category: str = "legal") -> int:
        """Create a new chat conversation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (title, category) VALUES (?, ?)",
            (title, category)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_conversations(self, category: str = None) -> List[Dict]:
        """Get all conversations, optionally filtered by category."""
        cursor = self.conn.cursor()
        if category:
            cursor.execute(
                "SELECT * FROM conversations WHERE category = ? ORDER BY updated_at DESC",
                (category,)
            )
        else:
            cursor.execute(
                "SELECT * FROM conversations ORDER BY is_pinned DESC, updated_at DESC"
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_conversation_messages(self, conversation_id: int) -> List[Dict]:
        """Get all messages for a conversation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_message(self, conversation_id: int, role: str, content: str,
                    tokens_used: int = 0, model: str = "") -> int:
        """Add a message to a conversation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, tokens_used, model) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, tokens_used, model)
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()

    def toggle_pin_conversation(self, conversation_id: int):
        """Toggle pin status of a conversation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE conversations SET is_pinned = CASE WHEN is_pinned = 1 THEN 0 ELSE 1 END WHERE id = ?",
            (conversation_id,)
        )
        self.conn.commit()

    # ==================== Document Methods ====================

    def add_document(self, filename: str, filepath: str, file_type: str,
                     file_size: int = 0, content: str = None) -> int:
        """Add a document record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO documents (filename, filepath, file_type, file_size, content) VALUES (?, ?, ?, ?, ?)",
            (filename, filepath, file_type, file_size, content)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_documents(self) -> List[Dict]:
        """Get all documents."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def update_document_analysis(self, doc_id: int, summary: str, metadata: dict = None):
        """Update document with analysis results."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE documents SET summary = ?, metadata = ?, analyzed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (summary, json.dumps(metadata or {}, ensure_ascii=False), doc_id)
        )
        self.conn.commit()

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get a single document by ID."""
        cursor = self._execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: int):
        """Delete a document record."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()

    # ==================== Case Methods ====================

    def create_case(self, title: str, case_number: str = None,
                    description: str = "", category: str = "civil",
                    priority: str = "medium") -> int:
        """Create a new legal case."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO cases (title, case_number, description, category, priority) VALUES (?, ?, ?, ?, ?)",
            (title, case_number, description, category, priority)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_cases(self, status: str = None) -> List[Dict]:
        """Get all cases, optionally filtered by status."""
        cursor = self.conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY updated_at DESC",
                (status,)
            )
        else:
            cursor.execute("SELECT * FROM cases ORDER BY updated_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def update_case_status(self, case_id: int, status: str):
        """Update case status."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE cases SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, case_id)
        )
        self.conn.commit()

    # ==================== Financial Methods ====================

    def add_financial_record(self, record_type: str, amount: float,
                             description: str = "", date: str = None,
                             case_id: int = None, currency: str = "IRR") -> int:
        """Add a financial record."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO financial_records (case_id, record_type, amount, currency, description, date) VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, record_type, amount, currency, description, date)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_financial_summary(self) -> Dict[str, float]:
        """Get financial summary statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT record_type, SUM(amount) as total FROM financial_records GROUP BY record_type")
        result = {"income": 0.0, "expense": 0.0, "settlement": 0.0, "fee": 0.0}
        for row in cursor.fetchall():
            result[row["record_type"]] = row["total"] or 0
        result["balance"] = result["income"] - result["expense"] - result["fee"]
        return result

    def get_financial_records(self, case_id: int = None) -> List[Dict]:
        """Get financial records, optionally filtered by case."""
        cursor = self.conn.cursor()
        if case_id:
            cursor.execute(
                "SELECT * FROM financial_records WHERE case_id = ? ORDER BY date DESC",
                (case_id,)
            )
        else:
            cursor.execute("SELECT * FROM financial_records ORDER BY date DESC")
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Risk Assessment Methods ====================

    def add_risk_assessment(self, title: str, description: str = "",
                            risk_level: str = "medium", risk_score: float = 0.0,
                            factors: list = None, recommendations: list = None,
                            case_id: int = None) -> int:
        """Add a risk assessment."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO risk_assessments (title, description, risk_level, risk_score, factors, recommendations, case_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, description, risk_level, risk_score,
             json.dumps(factors or [], ensure_ascii=False),
             json.dumps(recommendations or [], ensure_ascii=False),
             case_id)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_risk_assessments(self, case_id: int = None) -> List[Dict]:
        """Get risk assessments."""
        cursor = self.conn.cursor()
        if case_id:
            cursor.execute(
                "SELECT * FROM risk_assessments WHERE case_id = ? ORDER BY created_at DESC",
                (case_id,)
            )
        else:
            cursor.execute("SELECT * FROM risk_assessments ORDER BY created_at DESC")
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d["factors"] = json.loads(d["factors"])
            d["recommendations"] = json.loads(d["recommendations"])
            results.append(d)
        return results

    # ==================== Analytics Methods ====================

    def log_action(self, action: str, details: dict = None):
        """Log a user action for analytics."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO analytics (action, details) VALUES (?, ?)",
            (action, json.dumps(details or {}, ensure_ascii=False))
        )
        self.conn.commit()

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get usage analytics summary."""
        cursor = self.conn.cursor()

        # Total conversations
        cursor.execute("SELECT COUNT(*) as count FROM conversations")
        total_conversations = cursor.fetchone()["count"]

        # Total messages
        cursor.execute("SELECT COUNT(*) as count FROM messages WHERE role = 'user'")
        total_messages = cursor.fetchone()["count"]

        # Total documents
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        total_documents = cursor.fetchone()["count"]

        # Total cases
        cursor.execute("SELECT COUNT(*) as count FROM cases")
        total_cases = cursor.fetchone()["count"]

        # Active cases
        cursor.execute("SELECT COUNT(*) as count FROM cases WHERE status = 'active'")
        active_cases = cursor.fetchone()["count"]

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_documents": total_documents,
            "total_cases": total_cases,
            "active_cases": active_cases,
        }

    def _execute(self, query, params=None):
        with self._lock:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor

    def close(self):
        """Close the database connection."""
        with self._lock:
            self.conn.close()


# Singleton instance
_db_instance = None


def get_database() -> DatabaseManager:
    """Get the global DatabaseManager instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
