"""Package metadata and public API."""

from memwatch.core import (
    MemoryEntry,
    StaleReport,
    analyze,
    detect_contradictions,
    find_duplicates,
    parse_store,
    parse_json_store,
    parse_sqlite_store,
    _parse_jsonl_store,
)

__all__ = [
    "MemoryEntry",
    "StaleReport",
    "analyze",
    "detect_contradictions",
    "find_duplicates",
    "parse_store",
    "parse_json_store",
    "parse_sqlite_store",
    "_parse_jsonl_store",
]
