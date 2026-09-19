"""Tests for memwatch core functionality."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from memwatch.core import (
    MemoryEntry,
    analyze,
    detect_contradictions,
    find_duplicates,
    parse_json_store,
    parse_sqlite_store,
    _age_score,
    _confirmation_score,
    _are_contradictory,
    _jaccard,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_entries():
    now = datetime.now(timezone.utc)
    return [
        MemoryEntry(id="e1", content="Uses jsonwebtoken for auth", created_at=now, confirmed_count=3),
        MemoryEntry(id="e2", content="Uses jose library for JWT auth", created_at=now - timedelta(days=60), confirmed_count=0),
        MemoryEntry(id="e3", content="Prefers TypeScript for new projects", created_at=now - timedelta(days=120), confirmed_count=0),
        MemoryEntry(id="e4", content="Prefers JavaScript not TypeScript for new projects", created_at=now, confirmed_count=1),
        MemoryEntry(id="e5", content="Uses jsonwebtoken for auth", created_at=now - timedelta(days=5), confirmed_count=1),  # duplicate of e1
        MemoryEntry(id="e6", content="Database is PostgreSQL", created_at=now - timedelta(days=200), confirmed_count=0),
    ]


@pytest.fixture
def json_store(tmp_path):
    data = {
        "facts": [
            {"id": "f1", "content": "Uses pydantic for validation", "created_at": "2026-01-15T10:00:00Z", "confirmed_count": 5},
            {"id": "f2", "content": "Uses marshmallow for serialization", "created_at": "2025-11-01T10:00:00Z", "confirmed_count": 0},
            {"id": "f3", "content": "Uses pydantic for validation", "created_at": "2026-02-01T10:00:00Z", "confirmed_count": 2},
        ]
    }
    path = tmp_path / "memories.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sqlite_store(tmp_path):
    db_path = tmp_path / "memories.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            created_at TEXT,
            confirmed_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?)", ("s1", "Uses Redis for caching", "2026-02-01T10:00:00+00:00", 4))
    conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?)", ("s2", "Uses Memcached for caching", "2025-10-01T10:00:00+00:00", 0))
    conn.commit()
    conn.close()
    return str(db_path)


# ── Unit tests ───────────────────────────────────────────────────────────

class TestStaleScoring:
    def test_fresh_entry(self):
        entry = MemoryEntry(id="x", content="test", created_at=datetime.now(timezone.utc), confirmed_count=5)
        score = _age_score(entry)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_half_life(self):
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(id="x", content="test", created_at=now - timedelta(days=30), confirmed_count=0)
        score = _age_score(entry, half_life_days=30)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_very_old(self):
        entry = MemoryEntry(id="x", content="test", created_at=datetime.now(timezone.utc) - timedelta(days=365), confirmed_count=0)
        score = _age_score(entry)
        assert score > 0.99

    def test_unconfirmed_stale(self):
        entry = MemoryEntry(id="x", content="test", created_at=datetime.now(timezone.utc), confirmed_count=0)
        score = _confirmation_score(entry)
        assert score == pytest.approx(0.7, abs=0.01)

    def test_well_confirmed(self):
        entry = MemoryEntry(id="x", content="test", created_at=datetime.now(timezone.utc), confirmed_count=10)
        score = _confirmation_score(entry)
        assert score == 0.0


class TestContradictionDetection:
    def test_direct_negation(self):
        assert _are_contradictory("uses jsonwebtoken", "doesn't use jsonwebtoken")

    def test_no_contradiction(self):
        assert not _are_contradictory("uses postgres", "uses redis")

    def test_overlapping_subject(self):
        assert _are_contradictory("prefers typescript for new code", "prefers javascript not typescript for new code")


class TestDeduplication:
    def test_high_overlap(self):
        assert _jaccard("uses pydantic for validation", "uses pydantic for validation") == 1.0

    def test_partial_overlap(self):
        score = _jaccard("uses redis for caching", "uses memcached for caching")
        assert 0.3 < score < 0.8

    def test_no_overlap(self):
        assert _jaccard("completely different words", "nothing alike here") == 0.0


class TestParsers:
    def test_json_parser(self, json_store):
        entries = parse_json_store(json_store)
        assert len(entries) == 3
        assert entries[0].id == "f1"
        assert entries[0].confirmed_count == 5

    def test_sqlite_parser(self, sqlite_store):
        entries = parse_sqlite_store(sqlite_store)
        assert len(entries) == 2
        assert entries[0].id == "s1"
    
    def test_sqlite_parser_rejects_injected_table_name(self, sqlite_store):
        malicious_table_names = [
            "memories; DROP TABLE memories; --",
            "memories UNION SELECT * FROM users",
            "memories --",
            ""
        ]
        for bad_table in malicious_table_names:
            with pytest.raises(ValueError, match="valid SQLite identifier"):
                parse_sqlite_store("dummy_path.db", table=bad_table)
        #Successfull parsing of a valid database
        assert len(parse_sqlite_store(sqlite_store)) == 2


# ── Integration tests ────────────────────────────────────────────────────

class TestAnalyze:
    def test_full_analysis(self, json_store):
        result = analyze(json_store)
        assert result["total_entries"] == 3
        assert "stats" in result
        assert "flagged" in result

    def test_contradictions_detected(self):
        """Entries with negation pairs should be flagged."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"facts": [
                {"id": "a", "content": "uses postgres for database", "created_at": "2026-01-01T00:00:00Z"},
                {"id": "b", "content": "doesn't use postgres, uses mysql", "created_at": "2026-01-01T00:00:00Z"},
            ]}, f)
            f.flush()
            result = analyze(f.name)
            # At least one should have contradictions flagged
            flagged_with_contradictions = [r for r in result["flagged"] if r.contradictions]
            assert len(flagged_with_contradictions) >= 1

    def test_duplicates_detected(self):
        """Duplicate entries should be flagged."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"facts": [
                {"id": "x", "content": "uses pydantic for data validation", "created_at": "2026-01-01T00:00:00Z"},
                {"id": "y", "content": "uses pydantic for data validation", "created_at": "2026-01-01T00:00:00Z"},
            ]}, f)
            f.flush()
            result = analyze(f.name)
            flagged_with_dups = [r for r in result["flagged"] if r.duplicates]
            assert len(flagged_with_dups) >= 1

    def test_old_unconfirmed_flagged(self):
        """Old + unconfirmed entries should be flagged."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"facts": [
                {"id": "old", "content": "legacy fact from long ago", "created_at": old_date, "confirmed_count": 0},
            ]}, f)
            f.flush()
            result = analyze(f.name)
            assert len(result["flagged"]) >= 1
            assert result["reports"][0].stale_score > 0.5
