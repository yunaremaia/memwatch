"""Tests for SQL injection prevention in parse_sqlite_store."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from memwatch.core import parse_sqlite_store, SAFE_TABLE_RE


class TestSQLInjectionPrevention:
    """Tests for memwatch#36: validate table name against SQL injection."""

    def test_safe_table_regex_accepts_valid(self) -> None:
        assert SAFE_TABLE_RE.match("memories")
        assert SAFE_TABLE_RE.match("entries")
        assert SAFE_TABLE_RE.match("_temp")
        assert SAFE_TABLE_RE.match("table123")
        assert SAFE_TABLE_RE.match("a")

    def test_safe_table_regex_rejects_injection(self) -> None:
        # SQL injection attempts
        assert not SAFE_TABLE_RE.match("entries; DROP TABLE memories;--")
        assert not SAFE_TABLE_RE.match("' OR '1'='1")
        assert not SAFE_TABLE_RE.match("table;--")
        assert not SAFE_TABLE_RE.match("1=1")
        # Path traversal
        assert not SAFE_TABLE_RE.match("../../etc/passwd")
        # Space-based injection
        assert not SAFE_TABLE_RE.match("entries WHERE 1=1")
        # Empty
        assert not SAFE_TABLE_RE.match("")
        # Special chars
        assert not SAFE_TABLE_RE.match("table name")
        assert not SAFE_TABLE_RE.match("table\nname")

    def test_valid_table_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT, content TEXT)")
        conn.execute("INSERT INTO memories VALUES ('1', 'test')")
        conn.commit()
        conn.close()

        entries = parse_sqlite_store(str(db_path), "memories")
        assert len(entries) == 1
        assert entries[0].content == "test"

    def test_invalid_table_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT, content TEXT)")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="Invalid table name"):
            parse_sqlite_store(str(db_path), "entries; DROP TABLE memories;--")

    def test_sql_injection_attempt_does_not_execute(self, tmp_path: Path) -> None:
        """Even if injection string somehow bypasses regex, table doesn't exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT, content TEXT)")
        conn.execute("INSERT INTO memories VALUES ('1', 'secret')")
        conn.commit()
        conn.close()

        # Attempt injection — should raise ValueError, not execute
        with pytest.raises(ValueError):
            parse_sqlite_store(str(db_path), "memories; DELETE FROM memories;--")

        # Verify data is intact
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        assert count == 1  # No rows deleted
