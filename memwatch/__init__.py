"""Package metadata and public API."""

from memwatch.core import (
    MemoryEntry,
    StaleReport,
    _parse_jsonl_store,
    analyze,
    detect_contradictions,
    find_duplicates,
    parse_json_store,
    parse_sqlite_store,
    parse_store,
    parse_yaml_store,
)

__all__ = [
    "MemoryEntry",
    "StaleReport",
    "_parse_jsonl_store",
    "analyze",
    "detect_contradictions",
    "find_duplicates",
    "parse_json_store",
    "parse_sqlite_store",
    "parse_store",
    "parse_yaml_store",
]
