"""Memory health monitor for AI agent persistent stores."""

__version__ = "0.1.0"

import json
import logging
import re

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Data model ───────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory fact/episode from an agent store."""
    id: str
    content: str
    created_at: datetime
    confirmed_count: int = 0
    last_confirmed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 86400


@dataclass
class ParseResult:
    """Parsed entries together with details about records that were skipped."""
    entries: list[MemoryEntry]
    skipped_entry_details: list[dict[str, str]] = field(default_factory=list)


@dataclass
class StaleReport:
    """Result of staleness analysis for one entry."""
    entry: MemoryEntry
    stale_score: float  # 0.0 (fresh) → 1.0 (rotten)
    reasons: list[str]
    duplicates: list[MemoryEntry] = field(default_factory=list)
    contradictions: list[MemoryEntry] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        return self.stale_score > 0.6 or bool(self.contradictions)

    @property
    def action(self) -> str:
        if self.stale_score > 0.8:
            return "DELETE"
        if self.contradictions:
            return "REVIEW"
        if self.duplicates:
            return "MERGE"
        if self.stale_score > 0.6:
            return "REFRESH"
        return "KEEP"


# ── Stale scoring engine ─────────────────────────────────────────────────

NEGATION_PAIRS = [
    ("uses", "doesn't use"), ("is", "is not"), ("prefer", "don't prefer"),
    ("enabled", "disabled"), ("true", "false"), ("yes", "no"),
    ("always", "never"), ("increased", "decreased"),
]

def _age_score(entry: MemoryEntry, half_life_days: float = 30.0) -> float:
    """Exponential decay of freshness: score 0.5 at half_life."""
    import math
    return 1.0 - math.exp(-0.693 * entry.age_days / half_life_days)


def _confirmation_score(entry: MemoryEntry) -> float:
    """More confirmations → less stale. Diminishing returns after 5."""
    if entry.confirmed_count == 0:
        return 0.7  # unconfirmed = moderately stale
    return max(0.0, 0.5 - entry.confirmed_count * 0.1)


def compute_stale_score(entry: MemoryEntry, all_entries: list[MemoryEntry]) -> StaleReport:
    """Compute composite stale score with reasons."""
    age = _age_score(entry)
    conf = _confirmation_score(entry)
    score = 0.6 * age + 0.4 * conf

    reasons = []
    if entry.age_days > 90:
        reasons.append(f"Old: {entry.age_days:.0f} days")
    if entry.confirmed_count == 0:
        reasons.append("Never confirmed")
    if entry.age_days > 30 and entry.confirmed_count < 2:
        reasons.append("Stale + unconfirmed")

    return StaleReport(entry=entry, stale_score=round(score, 3), reasons=reasons)


# ── Contradiction detection ───────────────────────────────────────────────

def detect_contradictions(entries: list[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    """Find pairs that contradict each other via negation patterns.

    Auto-switches to inverted index for stores >500 entries (O(n) vs O(n²)).
    """
    use_index = len(entries) > 500
    if not use_index:
        return _detect_contradictions_scan(entries)
    return _detect_contradictions_index(entries)


def _detect_contradictions_scan(entries: list[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    """O(n²) brute-force scan — accurate for small stores."""
    contradictions: dict[str, list[MemoryEntry]] = {}
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if _are_contradictory(a.content, b.content):
                contradictions.setdefault(a.id, []).append(b)
                contradictions.setdefault(b.id, []).append(a)
    return contradictions


class ContradictionIndex:
    """Inverted index for O(n) contradiction candidate lookup.

    Builds a token → entry_id index in O(n), then checks only entries
    sharing at least one token as contradiction candidates.
    """

    def __init__(self, entries: list[MemoryEntry]):
        self.entries = {e.id: e for e in entries}
        self.token_index: dict[str, set[str]] = {}
        self._build(entries)

    def _build(self, entries: list[MemoryEntry]) -> None:
        """Build inverted index: normalized token → set of entry IDs."""
        for entry in entries:
            for token in _tokenize(entry.content):
                self.token_index.setdefault(token, set()).add(entry.id)

    def find_candidates(self) -> set[tuple[str, str]]:
        """Return candidate pairs (id_a, id_b) that share tokens."""
        candidates: set[tuple[str, str]] = set()
        for ids in self.token_index.values():
            if len(ids) < 2:
                continue
            id_list = sorted(ids)
            for i, a in enumerate(id_list):
                for b in id_list[i + 1:]:
                    candidates.add((a, b))
        return candidates

    def find_contradictions(self) -> dict[str, list[MemoryEntry]]:
        """Find contradictions using candidate pairs from the index."""
        contradictions: dict[str, list[MemoryEntry]] = {}
        for id_a, id_b in self.find_candidates():
            a, b = self.entries[id_a], self.entries[id_b]
            if _are_contradictory(a.content, b.content):
                contradictions.setdefault(a.id, []).append(b)
                contradictions.setdefault(b.id, []).append(a)
        return contradictions


def _detect_contradictions_index(entries: list[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    """O(n) inverted-index detection for large stores."""
    index = ContradictionIndex(entries)
    return index.find_contradictions()


def _tokenize(text: str) -> set[str]:
    """Tokenize text into normalized words for indexing."""
    return {w for w in text.lower().split() if len(w) >= 3}


def _normalize(text: str) -> set[str]:
    """Normalize text for comparison: lowercase, strip punctuation, handle contractions.

    Stemming is deliberately conservative: a trailing ``s`` is stripped only
    for words longer than 3 characters that do not end in ``ss``/``us``/``is``
    (the common non-plural suffixes). This keeps genuine plural collisions
    (``tests``→``test``, ``runs``→``run``) while no longer mangling words whose
    final ``s`` is part of the stem — ``address`` and ``process`` used to lose
    a character and fail to match themselves, and words like ``iris`` or
    ``bonus`` are left intact (issue #63).
    """
    import re
    t = text.lower()
    t = t.replace("n't", " not").replace("'s", " is").replace("'re", " are")
    t = re.sub(r'[^a-z0-9\s]', '', t)
    words = set(t.split())
    return {_strip_plural(w) for w in words}


def _strip_plural(w: str) -> str:
    """Strip a trailing plural ``s`` only when it is safe to do so."""
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _are_contradictory(text_a: str, text_b: str) -> bool:
    """Simple heuristic: shared subject + negation flip."""
    a = text_a.lower().strip()
    b = text_b.lower().strip()
    na = _normalize(a)
    nb = _normalize(b)
    
    for pos, neg in NEGATION_PAIRS:
        if (pos in a and neg in b) or (neg in a and pos in b):
            overlap = na & nb
            if len(overlap) >= 2:
                return True
        if (f"not {pos}" in b and pos in a) or (f"not {pos}" in a and pos in b):
            overlap = na & nb
            if len(overlap) >= 2:
                return True
    
    words_a = a.split()
    words_b = b.split()
    
    not_words_a = {words_a[i+1] for i, w in enumerate(words_a) if w == "not" and i+1 < len(words_a)}
    not_words_b = {words_b[i+1] for i, w in enumerate(words_b) if w == "not" and i+1 < len(words_b)}
    
    for nw in not_words_a:
        if nw in set(words_b):
            return True
    for nw in not_words_b:
        if nw in set(words_a):
            return True
    
    return False


# ── Deduplication ─────────────────────────────────────────────────────────

def find_duplicates(entries: list[MemoryEntry], threshold: float = 0.85) -> dict[str, list[MemoryEntry]]:
    """Group entries with high word-overlap."""
    from collections import defaultdict
    groups: dict[str, list[MemoryEntry]] = defaultdict(list)
    seen = set()
    for i, a in enumerate(entries):
        if a.id in seen:
            continue
        group = [a]
        for b in entries[i+1:]:
            if b.id in seen:
                continue
            if _jaccard(a.content, b.content) >= threshold:
                group.append(b)
                seen.add(b.id)
        if len(group) > 1:
            seen.add(a.id)
            groups[a.id] = group
    return groups


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ── Parsers ──────────────────────────────────────────────────────────────

def _is_jsonl(path: str) -> bool:
    """Detect JSONL format: file has multiple non-empty lines, each parseable as JSON."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        return False
    jsonl_count = 0
    for line in lines[:5]:
        if line.startswith('{') or line.startswith('['):
            try:
                json.loads(line)
                jsonl_count += 1
            except json.JSONDecodeError:
                pass
    return jsonl_count >= 2


def _parse_valid_timestamp(value: Any) -> datetime | None:
    """Return a parsed timestamp, or ``None`` when a supplied value is invalid."""
    if value is None or value == "":
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    from datetime import date
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _skipped_entry(location: str, reason: str) -> dict[str, str]:
    logger.warning("Skipping %s: %s", location, reason)
    return {"location": location, "reason": reason}


def _parse_json_entry(
    item: Any,
    location: str,
    fallback_id: str | None = None,
) -> tuple[MemoryEntry | None, dict[str, str] | None]:
    """Validate and parse one JSON entry without aborting the whole store."""
    if isinstance(item, str):
        item = {"id": item, "content": item}
    if not isinstance(item, dict):
        return None, _skipped_entry(location, "entry is not an object")

    content = item.get("content", item.get("text", item.get("value", "")))
    if not isinstance(content, str):
        return None, _skipped_entry(location, "content is not a string")

    confirmed = item.get("confirmed_count", item.get("confirmations", 0))
    if isinstance(confirmed, str):
        try:
            confirmed = int(confirmed)
        except ValueError:
            return None, _skipped_entry(
                location, "confirmed_count is not a valid integer"
            )
    elif isinstance(confirmed, bool) or not isinstance(confirmed, int):
        return None, _skipped_entry(
            location, "confirmed_count has unexpected type"
        )

    created_at = _parse_valid_timestamp(
        item.get("created_at", item.get("timestamp", ""))
    )
    if created_at is None:
        return None, _skipped_entry(location, "created_at is not a valid timestamp")

    entry_id = item.get("id", item.get("key"))
    if entry_id is None:
        entry_id = fallback_id if fallback_id is not None else str(hash(content))

    meta = {
        k: v for k, v in item.items()
        if k not in ("id", "content", "text", "value", "created_at", "timestamp")
    }
    if isinstance(item.get("metadata"), dict):
        meta.update(item["metadata"])

    entry = MemoryEntry(
        id=str(entry_id),
        content=content,
        created_at=created_at,
        confirmed_count=confirmed,
        last_confirmed_at=_parse_ts(item.get("last_confirmed_at"))
        if item.get("last_confirmed_at") else None,
        metadata=meta,
    )
    return entry, None


def _parse_json_store_result(path: str, fmt: str = "auto") -> ParseResult:
    """Parse a JSON store and retain details about invalid entries."""
    if fmt == "auto":
        fmt = "jsonl" if _is_jsonl(path) else "json"

    if fmt == "jsonl":
        return _parse_jsonl_store_result(path)

    with open(path) as f:
        data = json.load(f)
    entries = []
    skipped_entry_details = []
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("facts", "episodes", "memories", "entries"):
            if key in data:
                items.extend(data[key])
        if not items:
            items = [
                {"id": k, **v} if isinstance(v, dict) else {"id": k, "content": str(v)}
                for k, v in data.items()
            ]

    for entry_num, item in enumerate(items, 1):
        entry, skipped = _parse_json_entry(item, f"entry {entry_num}")
        if entry is not None:
            entries.append(entry)
        if skipped is not None:
            skipped_entry_details.append(skipped)
    return ParseResult(entries, skipped_entry_details)


def parse_json_store(path: str, fmt: str = "auto") -> list[MemoryEntry]:
    """Parse a JSON memory store. Auto-detects JSON vs JSONL format.

    Supports:
    - Single JSON document (list or dict with known keys)
    - JSONL format (one JSON object per line)

    Args:
        path: Path to the JSON/JSONL file
        fmt: Format to parse — 'json', 'jsonl', or 'auto' (default: auto-detect)
    """
    return _parse_json_store_result(path, fmt=fmt).entries


def _parse_jsonl_store(path: str) -> list[MemoryEntry]:
    """Parse a JSONL memory store (one JSON object per line)."""
    return _parse_jsonl_store_result(path).entries


def _parse_jsonl_store_result(path: str) -> ParseResult:
    """Parse a JSONL store and retain line-level skip details."""
    entries = []
    skipped_entry_details = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                skipped_entry_details.append(
                    _skipped_entry(f"line {line_num}", f"malformed JSON: {e}")
                )
                continue

            entry, skipped = _parse_json_entry(
                item,
                f"line {line_num}",
                fallback_id=f"line_{line_num}",
            )
            if entry is not None:
                entries.append(entry)
            if skipped is not None:
                skipped_entry_details.append(skipped)
    return ParseResult(entries, skipped_entry_details)


SAFE_TABLE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def parse_sqlite_store(path: str, table: str = "memories") -> list[MemoryEntry]:
    """Parse a SQLite memory store."""
    if not SAFE_TABLE_RE.match(table):
        raise ValueError(f"Invalid table name: {table}")
    import sqlite3
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(f"SELECT * FROM {table}")
    entries = []
    for row in cursor:
        d = dict(row)
        entry = MemoryEntry(
            id=str(d.get("id", d.get("key", hash(str(d.get("content", "")))))),
            content=d.get("content", d.get("text", d.get("value", ""))),
            created_at=_parse_ts(d.get("created_at", d.get("timestamp", ""))),
            confirmed_count=int(d.get("confirmed_count", d.get("confirmations", 0))),
            last_confirmed_at=_parse_ts(d.get("last_confirmed_at")) if d.get("last_confirmed_at") else None,
            metadata=d,
        )
        entries.append(entry)
    conn.close()
    return entries


def parse_yaml_store(content_or_path: str) -> list[MemoryEntry]:
    """Parse a YAML memory store.

    Supports:
    - Single-document YAML (list of entries, container with facts/memories, or single entry)
    - Multi-document YAML (separated by ---)
    - File path or raw YAML content string

    Args:
        content_or_path: Path to the YAML file or raw YAML content string.

    Returns:
        List of parsed MemoryEntry objects.
    """
    return _parse_yaml_store_result(content_or_path).entries


def _parse_yaml_store_result(content_or_path: str) -> ParseResult:
    """Parse a YAML memory store and retain details about skipped entries."""
    import yaml
    from pathlib import Path

    raw_content = ""
    source_name = "yaml"

    if isinstance(content_or_path, str) and "\n" not in content_or_path:
        try:
            p = Path(content_or_path)
            if p.is_file():
                source_name = str(content_or_path)
                with open(content_or_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            else:
                raw_content = content_or_path
        except (OSError, ValueError):
            raw_content = content_or_path
    else:
        raw_content = str(content_or_path)

    if not raw_content.strip():
        return ParseResult([], [])

    entries: list[MemoryEntry] = []
    skipped_entry_details: list[dict[str, str]] = []

    try:
        docs = list(yaml.safe_load_all(raw_content))
    except yaml.YAMLError as e:
        logger.warning("Malformed YAML in %s: %s", source_name, e)
        return ParseResult([], [_skipped_entry(source_name, f"malformed YAML: {e}")])

    def _normalize_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _normalize_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_normalize_keys(elem) for elem in obj]
        return obj

    for doc_idx, doc in enumerate(docs, 1):
        if doc is None:
            continue
        doc = _normalize_keys(doc)
        doc_prefix = f"doc {doc_idx}" if len(docs) > 1 else "entry"

        if isinstance(doc, list):
            for item_idx, item in enumerate(doc, 1):
                loc = f"{doc_prefix} item {item_idx}" if len(docs) > 1 else f"entry {item_idx}"
                entry, skipped = _parse_json_entry(item, loc, fallback_id=f"entry_{len(entries) + 1}")
                if entry is not None:
                    entries.append(entry)
                if skipped is not None:
                    skipped_entry_details.append(skipped)

        elif isinstance(doc, dict):
            found_container = False
            for key in ("facts", "episodes", "memories", "entries"):
                if key in doc and isinstance(doc[key], list):
                    found_container = True
                    for item_idx, item in enumerate(doc[key], 1):
                        loc = f"{doc_prefix} {key}[{item_idx}]"
                        entry, skipped = _parse_json_entry(item, loc, fallback_id=f"{key}_{item_idx}")
                        if entry is not None:
                            entries.append(entry)
                        if skipped is not None:
                            skipped_entry_details.append(skipped)
                    break
                elif key in doc and isinstance(doc[key], dict):
                    found_container = True
                    for item_idx, (k, v) in enumerate(doc[key].items(), 1):
                        loc = f"{doc_prefix} {key}[{k}]"
                        if isinstance(v, dict):
                            item = {"id": str(k), **v}
                        else:
                            item = {"id": str(k), "content": str(v)}
                        entry, skipped = _parse_json_entry(item, loc, fallback_id=str(k))
                        if entry is not None:
                            entries.append(entry)
                        if skipped is not None:
                            skipped_entry_details.append(skipped)
                    break

            if not found_container:
                if any(k in doc for k in ("content", "text", "value")):
                    loc = doc_prefix
                    entry, skipped = _parse_json_entry(doc, loc, fallback_id=f"entry_{len(entries) + 1}")
                    if entry is not None:
                        entries.append(entry)
                    if skipped is not None:
                        skipped_entry_details.append(skipped)
                else:
                    for item_idx, (k, v) in enumerate(doc.items(), 1):
                        loc = f"{doc_prefix} item {k}"
                        if isinstance(v, dict):
                            item = {"id": str(k), **v}
                        else:
                            item = {"id": str(k), "content": str(v)}
                        entry, skipped = _parse_json_entry(item, loc, fallback_id=str(k))
                        if entry is not None:
                            entries.append(entry)
                        if skipped is not None:
                            skipped_entry_details.append(skipped)

        elif isinstance(doc, str):
            loc = doc_prefix
            entry, skipped = _parse_json_entry({"id": f"entry_{len(entries) + 1}", "content": doc}, loc)
            if entry is not None:
                entries.append(entry)
            if skipped is not None:
                skipped_entry_details.append(skipped)
        else:
            skipped_entry_details.append(_skipped_entry(doc_prefix, "document is not an object or list"))

    return ParseResult(entries, skipped_entry_details)


def parse_store(path: str, fmt: str = "auto") -> list[MemoryEntry]:
    """Auto-detect format and parse."""
    return _parse_store_result(path, fmt=fmt).entries


def _parse_store_result(path: str, fmt: str = "auto") -> ParseResult:
    """Auto-detect format and retain parser warnings for the analysis summary."""
    if fmt in ("yaml", "yml") or path.endswith(".yaml") or path.endswith(".yml"):
        return _parse_yaml_store_result(path)
    elif fmt == "jsonl" or path.endswith(".jsonl"):
        return _parse_json_store_result(path, fmt="jsonl")
    elif fmt == "json" or path.endswith(".json"):
        return _parse_json_store_result(path, fmt=fmt)
    elif fmt in ("sqlite", "db") or path.endswith(".db") or path.endswith(".sqlite"):
        return ParseResult(parse_sqlite_store(path))
    else:
        try:
            return _parse_json_store_result(path, fmt=fmt)
        except Exception:
            try:
                return _parse_yaml_store_result(path)
            except Exception:
                return ParseResult(parse_sqlite_store(path))


def _parse_ts(ts) -> datetime:
    """Best-effort timestamp parsing."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not ts:
        return datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


# ── Analysis pipeline ────────────────────────────────────────────────────

def analyze(path: str, stale_threshold: float = 0.6, fmt: str = "auto", use_index: bool | None = None) -> dict:
    """Run full analysis on a memory store."""
    parse_result = _parse_store_result(path, fmt=fmt)
    entries = parse_result.entries
    contradictions = detect_contradictions(entries)
    duplicates = find_duplicates(entries)
    reports = []
    for entry in entries:
        report = compute_stale_score(entry, entries)
        if entry.id in contradictions:
            report.contradictions = contradictions[entry.id]
            report.stale_score = min(1.0, report.stale_score + 0.2)
            report.reasons.append(f"{len(report.contradictions)} contradiction(s)")
        for group in duplicates.values():
            if any(e.id == entry.id for e in group):
                report.duplicates = [e for e in group if e.id != entry.id]
                if report.duplicates:
                    report.reasons.append(f"{len(report.duplicates)} duplicate(s)")
                break
        reports.append(report)

    return {
        "total_entries": len(entries),
        "skipped_entries": len(parse_result.skipped_entry_details),
        "skipped_entry_details": parse_result.skipped_entry_details,
        "flagged": [r for r in reports if r.needs_attention],
        "healthy": [r for r in reports if not r.needs_attention],
        "reports": reports,
        "stats": {
            "avg_stale": sum(r.stale_score for r in reports) / max(len(reports), 1),
            "high_risk": len([r for r in reports if r.stale_score > 0.8]),
            "with_contradictions": len([r for r in reports if r.contradictions]),
            "with_duplicates": len([r for r in reports if r.duplicates]),
        },
    }
