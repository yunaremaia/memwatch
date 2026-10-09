"""Tests for memwatch.cli — covers scan/fix/dashboard/report commands."""

import json

import yaml
from click.testing import CliRunner

from memwatch.cli import cli


def _make_store(tmp_path, fmt="json"):
    """Create a minimal memory store for testing."""
    entries = [
        {"id": "entry-001", "content": "User prefers dark mode", "timestamp": "2020-01-01T00:00:00Z"},
        {"id": "entry-002", "content": "User prefers light mode", "timestamp": "2020-01-02T00:00:00Z"},
        {"id": "entry-003", "content": "Project uses Python 3.12", "timestamp": "2025-01-01T00:00:00Z"},
    ]
    path = tmp_path / f"store.{fmt}"
    if fmt == "json":
        path.write_text(json.dumps(entries))
    elif fmt == "jsonl":
        path.write_text("\n".join(json.dumps(e) for e in entries))
    elif fmt in ("yaml", "yml"):
        path.write_text(yaml.dump(entries))
    return path


class TestScanCommand:
    def test_scan_text_format(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["scan", str(path)])
        assert result.exit_code == 0
        assert "memwatch" in result.output
        assert "Entries:" in result.output

    def test_scan_json_format(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["scan", str(path), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "store" in data
        assert "stats" in data
        assert "flagged" in data

    def test_scan_yaml_format(self, tmp_path):
        path = _make_store(tmp_path, "yaml")
        result = CliRunner().invoke(cli, ["scan", str(path), "--format", "yaml"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert "store" in data

    def test_scan_verbose(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["scan", str(path), "--verbose"])
        assert result.exit_code == 0
        assert "Content:" in result.output

    def test_scan_with_threshold(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["scan", str(path), "--threshold", "0.9"])
        assert result.exit_code == 0

    def test_scan_jsonl_format(self, tmp_path):
        path = _make_store(tmp_path, "jsonl")
        result = CliRunner().invoke(cli, ["scan", str(path)])
        assert result.exit_code == 0

    def test_scan_store_format_override(self, tmp_path):
        path = _make_store(tmp_path, "json")
        result = CliRunner().invoke(cli, ["scan", str(path), "--store-format", "json"])
        assert result.exit_code == 0

    def test_scan_no_flagged(self, tmp_path):
        entries = [{"id": "fresh-001", "content": "Recent fact", "timestamp": "2026-10-09T00:00:00Z"}]
        path = tmp_path / "store.json"
        path.write_text(json.dumps(entries))
        result = CliRunner().invoke(cli, ["scan", str(path)])
        assert result.exit_code == 0
        assert "healthy" in result.output.lower()


class TestFixCommand:
    def test_fix_dry_run(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["fix", str(path), "--dry-run"])
        assert result.exit_code == 0

    def test_fix_nothing_to_fix(self, tmp_path):
        entries = [{"id": "fresh-001", "content": "Recent fact", "timestamp": "2026-10-09T00:00:00Z"}]
        path = tmp_path / "store.json"
        path.write_text(json.dumps(entries))
        result = CliRunner().invoke(cli, ["fix", str(path)])
        assert result.exit_code == 0
        assert "Nothing to fix" in result.output

    def test_fix_with_threshold(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["fix", str(path), "--threshold", "0.9", "--dry-run"])
        assert result.exit_code == 0


class TestDashboardCommand:
    def test_dashboard_text(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["dashboard", str(path)])
        assert result.exit_code == 0
        assert "MEMWATCH" in result.output

    def test_dashboard_json(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["dashboard", str(path), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_entries" in data
        assert "stats" in data

    def test_dashboard_yaml(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["dashboard", str(path), "--format", "yaml"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert "total_entries" in data


class TestReportCommand:
    def test_report_json(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["report", str(path)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "store" in data

    def test_report_text(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["report", str(path), "--format", "text"])
        assert result.exit_code == 0

    def test_report_yaml(self, tmp_path):
        path = _make_store(tmp_path)
        result = CliRunner().invoke(cli, ["report", str(path), "--format", "yaml"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert "store" in data


class TestCliGroup:
    def test_version(self):
        result = CliRunner().invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "scan" in result.output
        assert "fix" in result.output
        assert "dashboard" in result.output
