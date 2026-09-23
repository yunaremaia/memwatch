"""Tests for YAML memory store parsing and analysis in memwatch."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from memwatch.cli import cli
from memwatch.core import (
    MemoryEntry,
    analyze,
    parse_store,
    parse_yaml_store,
    _parse_yaml_store_result,
)


# ── Standalone Tests matching Issue Acceptance Criteria ─────────────────────


def test_yaml_store_parsing(tmp_path):
    """Verifies that YAML memory stores are parsed correctly."""
    yaml_content = """
- id: "mem_1"
  content: "User prefers light theme"
  created_at: "2026-01-15T10:00:00Z"
  confirmed_count: 3
  metadata:
    source: "chat"
- id: "mem_2"
  content: "User works at Tech Corp"
  created_at: "2026-02-01T12:00:00Z"
  confirmed_count: 5
  metadata:
    category: "employment"
"""
    p = tmp_path / "store.yaml"
    p.write_text(yaml_content)

    entries = parse_yaml_store(str(p))
    assert len(entries) == 2

    assert entries[0].id == "mem_1"
    assert entries[0].content == "User prefers light theme"
    assert entries[0].confirmed_count == 3
    assert entries[0].metadata.get("source") == "chat"

    assert entries[1].id == "mem_2"
    assert entries[1].content == "User works at Tech Corp"
    assert entries[1].confirmed_count == 5
    assert entries[1].metadata.get("category") == "employment"


def test_yaml_staleness_detection(tmp_path):
    """Verifies that stale and contradictory memories are detected in YAML stores."""
    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=120)).isoformat()
    fresh_date = now.isoformat()

    yaml_content = f"""
facts:
  - id: "old_fact"
    content: "Uses Python 3.8"
    created_at: "{old_date}"
    confirmed_count: 0
  - id: "contradiction_a"
    content: "Feature is enabled"
    created_at: "{fresh_date}"
    confirmed_count: 1
  - id: "contradiction_b"
    content: "Feature is disabled"
    created_at: "{fresh_date}"
    confirmed_count: 1
"""
    p = tmp_path / "agent_memory.yaml"
    p.write_text(yaml_content)

    result = analyze(str(p))
    assert result["total_entries"] == 3

    # Check stale detection
    old_report = next(r for r in result["reports"] if r.entry.id == "old_fact")
    assert old_report.stale_score > 0.6
    assert any("Old:" in r or "Never confirmed" in r for r in old_report.reasons)

    # Check contradiction detection
    assert result["stats"]["with_contradictions"] >= 2
    contra_ids = {r.entry.id for r in result["reports"] if r.contradictions}
    assert "contradiction_a" in contra_ids
    assert "contradiction_b" in contra_ids


# ── Structure & Multi-Document Tests ─────────────────────────────────────────


class TestYamlStoreParsing:
    def test_single_document_list(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("""
- id: "1"
  content: "First memory"
- id: "2"
  content: "Second memory"
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) == 2
        assert [e.id for e in entries] == ["1", "2"]

    def test_single_document_single_entry(self, tmp_path):
        p = tmp_path / "single.yaml"
        p.write_text("""
id: "solo_1"
content: "Standalone memory entry"
timestamp: "2026-03-01T00:00:00Z"
confirmed_count: 2
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) == 1
        assert entries[0].id == "solo_1"
        assert entries[0].content == "Standalone memory entry"
        assert entries[0].confirmed_count == 2

    def test_single_document_mapping(self, tmp_path):
        p = tmp_path / "mapping.yaml"
        p.write_text("""
fact_1:
  content: "Custom key memory"
  confirmed_count: 4
fact_2: "Simple string content"
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) == 2
        id_map = {e.id: e for e in entries}
        assert id_map["fact_1"].content == "Custom key memory"
        assert id_map["fact_1"].confirmed_count == 4
        assert id_map["fact_2"].content == "Simple string content"

    def test_multi_document_yaml(self, tmp_path):
        p = tmp_path / "multi.yaml"
        p.write_text("""---
id: "doc_1"
content: "Memory from doc 1"
---
id: "doc_2"
content: "Memory from doc 2"
---
- id: "doc_3_a"
  content: "List memory inside doc 3"
- id: "doc_3_b"
  content: "Second list memory inside doc 3"
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) == 4
        ids = [e.id for e in entries]
        assert ids == ["doc_1", "doc_2", "doc_3_a", "doc_3_b"]

    def test_raw_string_input(self):
        content = """
- id: "raw_1"
  content: "Passed as raw string"
"""
        entries = parse_yaml_store(content)
        assert len(entries) == 1
        assert entries[0].id == "raw_1"
        assert entries[0].content == "Passed as raw string"

    def test_date_and_timestamps(self, tmp_path):
        p = tmp_path / "dates.yaml"
        p.write_text("""
- id: "d1"
  content: "ISO string timestamp"
  created_at: "2026-05-10T14:30:00Z"
- id: "d2"
  content: "Native YAML date"
  created_at: 2026-05-10
- id: "d3"
  content: "Integer epoch timestamp"
  created_at: 1700000000
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) == 3
        for e in entries:
            assert isinstance(e.created_at, datetime)
            assert e.created_at.tzinfo is not None

    def test_container_keys_episodes_and_memories(self, tmp_path):
        p = tmp_path / "containers.yaml"
        p.write_text("""
episodes:
  - id: "ep1"
    content: "Episode one"
memories:
  - id: "mem1"
    content: "Memory one"
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) >= 1
        assert any(e.id == "ep1" for e in entries)


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestYamlEdgeCases:
    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert parse_yaml_store(str(p)) == []

    def test_whitespace_and_comments_only(self, tmp_path):
        p = tmp_path / "comments.yaml"
        p.write_text("# This is just a comment\n\n   \n# Another comment\n")
        assert parse_yaml_store(str(p)) == []

    def test_malformed_yaml_does_not_crash(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("key: [unclosed list\n  bad indentation: - :")
        result = _parse_yaml_store_result(str(p))
        assert result.entries == []
        assert len(result.skipped_entry_details) == 1
        assert "malformed YAML" in result.skipped_entry_details[0]["reason"]

    def test_non_string_keys(self, tmp_path):
        p = tmp_path / "int_keys.yaml"
        p.write_text("""
101: "Direct string memory with int key"
102:
  content: "Nested memory with int key"
  confirmed_count: 5
""")
        entries = parse_yaml_store(str(p))
        assert len(entries) == 2
        id_set = {e.id for e in entries}
        assert "101" in id_set
        assert "102" in id_set

    def test_non_string_content_skipped(self, tmp_path):
        p = tmp_path / "invalid_content.yaml"
        p.write_text("""
- id: "valid"
  content: "Valid memory"
- id: "invalid"
  content: [1, 2, 3]
""")
        result = _parse_yaml_store_result(str(p))
        assert len(result.entries) == 1
        assert result.entries[0].id == "valid"
        assert len(result.skipped_entry_details) == 1
        assert "content is not a string" in result.skipped_entry_details[0]["reason"]


# ── CLI Integration ──────────────────────────────────────────────────────────


class TestYamlCLI:
    def test_cli_scan_auto_detect_yaml(self, tmp_path):
        p = tmp_path / "store.yaml"
        p.write_text("""
- id: "m1"
  content: "CLI test memory"
""")
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", str(p)])
        assert result.exit_code == 0
        assert "Entries: 1" in result.output
        assert "CLI test memory" in result.output or "Memory store is healthy" in result.output

    def test_cli_scan_store_format_yaml(self, tmp_path):
        p = tmp_path / "custom_ext.dat"
        p.write_text("""
- id: "m1"
  content: "Specified store format"
""")
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", str(p), "--store-format", "yaml"])
        assert result.exit_code == 0
        assert "Entries: 1" in result.output

    def test_cli_scan_format_yaml_output(self, tmp_path):
        p = tmp_path / "store.yml"
        p.write_text("""
- id: "m1"
  content: "YAML output test"
""")
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", str(p), "--format", "yaml"])
        assert result.exit_code == 0
        parsed_out = yaml.safe_load(result.output)
        assert parsed_out["store"] == str(p)
        assert "stats" in parsed_out

    def test_cli_dashboard_format_yaml(self, tmp_path):
        p = tmp_path / "store.yaml"
        p.write_text("""
- id: "m1"
  content: "Dashboard YAML"
""")
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", str(p), "--format", "yaml"])
        assert result.exit_code == 0
        parsed_out = yaml.safe_load(result.output)
        assert parsed_out["total_entries"] == 1


# ── Auto-Detect in parse_store ───────────────────────────────────────────────


class TestAutoDetect:
    def test_yaml_and_yml_extensions(self, tmp_path):
        p1 = tmp_path / "test.yaml"
        p1.write_text("- id: '1'\n  content: 'yaml ext'")
        p2 = tmp_path / "test.yml"
        p2.write_text("- id: '2'\n  content: 'yml ext'")

        entries1 = parse_store(str(p1))
        entries2 = parse_store(str(p2))

        assert len(entries1) == 1 and entries1[0].content == "yaml ext"
        assert len(entries2) == 1 and entries2[0].content == "yml ext"
