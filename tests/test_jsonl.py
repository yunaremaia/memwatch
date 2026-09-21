"""Tests for JSONL parsing support in memwatch."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memwatch.core import (
    MemoryEntry,
    _is_jsonl,
    _parse_jsonl_store,
    parse_json_store,
    parse_store,
)


# ── JSONL detection ───────────────────────────────────────────────────


class TestIsJsonl:
    def test_single_json_document_returns_false(self, tmp_path):
        """A single-line JSON object is NOT JSONL."""
        p = tmp_path / "single.json"
        p.write_text(json.dumps({"id": "1", "content": "hello"}))
        assert _is_jsonl(str(p)) is False

    def test_two_dimensionall_array_returns_false(self, tmp_path):
        """A single multi-line JSON document is NOT JSONL."""
        p = tmp_path / "array.json"
        p.write_text(json.dumps([{"id": "1"}, {"id": "2"}]))
        assert _is_jsonl(str(p)) is False

    def test_jsonl_objects_returns_true(self, tmp_path):
        """Multiple JSON objects, one per line IS JSONL."""
        p = tmp_path / "store.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "first"}),
                json.dumps({"id": "2", "content": "second"}),
                json.dumps({"id": "3", "content": "third"}),
            ])
        )
        assert _is_jsonl(str(p)) is True

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("")
        assert _is_jsonl(str(p)) is False

    def test_single_line_with_newline_returns_false(self, tmp_path):
        """Single line + trailing newline still just 1 non-empty line."""
        p = tmp_path / "single.json"
        p.write_text(json.dumps({"id": "1"}) + "\n")
        assert _is_jsonl(str(p)) is False

    def test_jsonl_extension_preserved(self, tmp_path):
        """Files with .jsonl are not detected by _is_jsonl alone (content-based)."""
        p = tmp_path / "store.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "a"}),
                json.dumps({"id": "2", "content": "b"}),
            ])
        )
        # _is_jsonl checks content, not extension
        assert _is_jsonl(str(p)) is True


# ── JSONL parsing ────────────────────────────────────────────────────


class TestParseJsonlStore:
    def test_basic_parsing(self, tmp_path):
        p = tmp_path / "store.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "first memory"}),
                json.dumps({"id": "2", "content": "second memory"}),
            ])
        )
        entries = parse_json_store(str(p), fmt="jsonl")
        assert len(entries) == 2
        assert entries[0].id == "1"
        assert entries[0].content == "first memory"
        assert entries[1].id == "2"

    def test_auto_detection(self, tmp_path):
        p = tmp_path / "auto.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "a", "content": "x", "created_at": "2026-01-01T00:00:00Z"}),
                json.dumps({"id": "b", "content": "y", "created_at": "2026-02-01T00:00:00Z"}),
                json.dumps({"id": "c", "content": "z", "created_at": "2026-03-01T00:00:00Z"}),
            ])
        )
        entries = parse_json_store(str(p))  # auto
        assert len(entries) == 3
        assert entries[0].content == "x"

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSONL lines are skipped with a warning."""
        p = tmp_path / "broken.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "good"}),
                "this is not json",
                json.dumps({"id": "2", "content": "also good"}),
            ])
        )
        entries = parse_json_store(str(p), fmt="jsonl")
        assert len(entries) == 2
        assert entries[0].content == "good"
        assert entries[1].content == "also good"

    def test_empty_lines_ignored(self, tmp_path):
        p = tmp_path / "gaps.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "a"}),
                "",
                "",
                json.dumps({"id": "2", "content": "b"}),
            ])
        )
        entries = parse_json_store(str(p), fmt="jsonl")
        assert len(entries) == 2



    def test_timestamps_parsed(self, tmp_path):
        p = tmp_path / "ts.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "x", "created_at": 1700000000}),
                json.dumps({"id": "2", "content": "y", "created_at": "2026-06-15T12:00:00Z"}),
            ])
        )
        entries = parse_json_store(str(p), fmt="jsonl")
        assert len(entries) == 2
        assert isinstance(entries[0].created_at, datetime)
        assert isinstance(entries[1].created_at, datetime)

    def test_confirmed_count_field(self, tmp_path):
        p = tmp_path / "confirmed.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "a", "confirmed_count": 5}),
                json.dumps({"id": "2", "content": "b", "confirmations": 3}),
            ])
        )
        entries = parse_json_store(str(p), fmt="jsonl")
        assert entries[0].confirmed_count == 5
        assert entries[1].confirmed_count == 3

    def test_schema_violation_warning_includes_line_number(self, tmp_path, caplog):
        p = tmp_path / "invalid.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "good"}),
                json.dumps({"id": "2", "content": None}),
                json.dumps({"id": "3", "content": "also good"}),
            ])
        )

        entries = parse_json_store(str(p), fmt="jsonl")

        assert [entry.id for entry in entries] == ["1", "3"]
        assert "line 2" in caplog.text
        assert "content is not a string" in caplog.text

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        entries = parse_json_store(str(p), fmt="jsonl")
        assert entries == []

    def test_large_jsonl_performance(self, tmp_path):
        """Parsing 1000 JSONL entries completes quickly."""
        p = tmp_path / "large.jsonl"
        lines = [
            json.dumps({"id": f"entry_{i}", "content": f"memory {i}"}) for i in range(1000)
        ]
        p.write_text("\n".join(lines))
        entries = parse_json_store(str(p), fmt="jsonl")
        assert len(entries) == 1000


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
    def test_json_document_still_works(self, tmp_path):
        """Original JSON document format still parses correctly."""
        p = tmp_path / "store.json"
        p.write_text(json.dumps([
            {"id": "1", "content": "hello"},
            {"id": "2", "content": "world"},
        ]))
        entries = parse_json_store(str(p), fmt="json")
        assert len(entries) == 2

    def test_json_with_facts_key(self, tmp_path):
        """JSON with 'facts' key still works."""
        p = tmp_path / "facts.json"
        p.write_text(json.dumps({
            "facts": [
                {"id": "f1", "content": "fact one"},
                {"id": "f2", "content": "fact two"},
            ]
        }))
        entries = parse_json_store(str(p), fmt="json")
        assert len(entries) == 2

    def test_store_format_auto_for_json(self, tmp_path):
        """parse_store auto-detects JSON vs JSONL by extension."""
        json_path = tmp_path / "store.json"
        json_path.write_text(json.dumps([{"id": "1", "content": "a"}]))

        jsonl_path = tmp_path / "store.jsonl"
        jsonl_path.write_text(
            "\n".join([
                json.dumps({"id": "1", "content": "x"}),
                json.dumps({"id": "2", "content": "y"}),
            ])
        )

        entries_json = parse_store(str(json_path))
        entries_jsonl = parse_store(str(jsonl_path))

        assert len(entries_json) == 1
        assert len(entries_jsonl) == 2

    def test_extension_jsonl_forces_jsonl_format(self, tmp_path):
        """A .jsonl file is forced to JSONL parsing even if content looks like JSON."""
        p = tmp_path / "weird.jsonl"
        # Single object with .jsonl extension — should still try jsonl
        p.write_text(json.dumps({"id": "1", "content": "hello"}))
        # This will be parsed as JSONL (one line = one object)
        entries = parse_store(str(p))
        assert len(entries) == 1
