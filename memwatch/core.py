"""Memory health monitor for AI agent persistent stores."""

__version__ = "0.1.0"

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
    """Find pairs that contradict each other via negation patterns."""
    contradictions: dict[str, list[MemoryEntry]] = {}
    for i, a in enumerate(entries):
        for b in entries[i+1:]:
            if _are_contradictory(a.content, b.content):
                contradictions.setdefault(a.id, []).append(b)
                contradictions.setdefault(b.id, []).append(a)
    return contradictions


def _normalize(text: str) -> set[str]:
    """Normalize text for comparison: lowercase, strip punctuation, handle contractions."""
    import re
    t = text.lower()
    # Expand common contractions
    t = t.replace("n't", " not").replace("'s", " is").replace("'re", " are")
    # Remove non-alphanumeric except spaces
    t = re.sub(r'[^a-z0-9\s]', '', t)
    words = set(t.split())
    # Simple stemming: remove trailing 's' for plurals/verbs
    return {w[:-1] if w.endswith('s') and len(w) > 3 else w for w in words}


def _are_contradictory(text_a: str, text_b: str) -> bool:
    """Simple heuristic: shared subject + negation flip."""
    a = text_a.lower().strip()
    b = text_b.lower().strip()
    na = _normalize(a)
    nb = _normalize(b)
    
    for pos, neg in NEGATION_PAIRS:
        # Check both directions: pos in a + neg in b, or neg in a + pos in b
        if (pos in a and neg in b) or (neg in a and pos in b):
            overlap = na & nb
            if len(overlap) >= 2:
                return True
        # Handle "not X" vs "X" pattern
        if (f"not {pos}" in b and pos in a) or (f"not {pos}" in a and pos in b):
            overlap = na & nb
            if len(overlap) >= 2:
                return True
    
    # Handle "not {word}" for shared words
    # Find words preceded by "not" in one text that appear normally in the other
    words_a = a.split()
    words_b = b.split()
    
    not_words_a = {words_a[i+1] for i, w in enumerate(words_a) if w == "not" and i+1 < len(words_a)}
    not_words_b = {words_b[i+1] for i, w in enumerate(words_b) if w == "not" and i+1 < len(words_b)}
    
    # If a word is "not X" in one and plain X in the other, it's a contradiction
    for nw in not_words_a:
        if nw in set(words_b):
            return True
    for nw in not_words_b:
        if nw in set(words_a):
            return True
    
    return False


# ── Deduplication ─────────────────────────────────────────────────────────

def find_duplicates(entries: list[MemoryEntry], threshold: float = 0.85) -> dict[str, list[MemoryEntry]]:
    """Group entries with high word-overlap (cheap, no embeddings needed)."""
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

def parse_json_store(path: str) -> list[MemoryEntry]:
    """Parse a JSON memory store (common format: {"facts": [...], "episodes": [...]})."""
    import json
    with open(path) as f:
        data = json.load(f)
    entries = []
    # Support multiple shapes
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("facts", "episodes", "memories", "entries"):
            if key in data:
                items.extend(data[key])
        if not items:
            items = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "content": str(v)}
                      for k, v in data.items()]

    for item in items:
        if isinstance(item, str):
            item = {"id": item, "content": item}
        entry = MemoryEntry(
            id=str(item.get("id", item.get("key", hash(item.get("content", ""))))),
            content=item.get("content", item.get("text", item.get("value", ""))),
            created_at=_parse_ts(item.get("created_at", item.get("timestamp", ""))),
            confirmed_count=int(item.get("confirmed_count", item.get("confirmations", 0))),
            last_confirmed_at=_parse_ts(item.get("last_confirmed_at")) if item.get("last_confirmed_at") else None,
            metadata={k: v for k, v in item.items() if k not in ("id", "content", "text", "value", "created_at", "timestamp")},
        )
        entries.append(entry)
    return entries

import re
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

def parse_sqlite_store(path: str, table: str = "memories") -> list[MemoryEntry]:
    """Parse a SQLite memory store using a validated table identifier."""

    if _TABLE_NAME_RE.fullmatch(table) is None:
        raise ValueError("table must be a valid SQLite identifier")
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


def parse_store(path: str) -> list[MemoryEntry]:
    """Auto-detect format and parse."""
    if path.endswith(".json"):
        return parse_json_store(path)
    elif path.endswith(".db") or path.endswith(".sqlite"):
        return parse_sqlite_store(path)
    else:
        # Try JSON first, then SQLite
        try:
            return parse_json_store(path)
        except Exception:
            return parse_sqlite_store(path)


def _parse_ts(ts) -> datetime:
    """Best-effort timestamp parsing."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not ts:
        return datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    # ISO format
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


# ── Analysis pipeline ────────────────────────────────────────────────────

def analyze(path: str, stale_threshold: float = 0.6) -> dict:
    """Run full analysis on a memory store."""
    entries = parse_store(path)
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
