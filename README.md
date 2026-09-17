# memwatch — Agent Memory Health Monitor

Monitor AI agent memory stores for rot, contradictions, and duplicates.

## The Problem

AI agents with persistent memory (Claude Code, Cursor, custom LLM agents) suffer from **memory rot**:
- Facts become stale (libraries change, preferences shift)
- Contradictions accumulate ("uses JWT" + "uses jose")
- Duplicates multiply (same decision stored 5 times)

`memwatch` scans your agent's memory store and tells you exactly what's rotten.

## Install

```bash
pip install memwatch
```

## Quick Start

```bash
# Scan a JSON memory store
memwatch scan path/to/memories.json

# See full details
memwatch scan path/to/memories.json --verbose

# JSON output for CI pipelines
memwatch scan path/to/memories.json --format json

# Get fix suggestions
memwatch fix path/to/memories.json --dry-run

# View dashboard
memwatch dashboard path/to/memories.json
```

## Supported Formats

- **JSON** — `{"facts": [{"id": ..., "content": ..., "created_at": ...}]}`
- **SQLite** — `memories` table with standard columns
- **Auto-detect** — pass any path, memwatch figures it out

## Scoring

Each memory entry gets a **stale score** (0.0 = fresh, 1.0 = rotten):

| Factor | Weight |
|--------|--------|
| Age (exponential decay, half-life 30 days) | 60% |
| Confirmation count | 40% |
| Contradictions found | +20% boost |

Actions suggested:
- **DELETE** — score > 0.8
- **REVIEW** — has contradictions
- **MERGE** — has duplicates
- **REFRESH** — score > 0.6, needs reconfirmation
- **KEEP** — healthy

## Python API

```python
from memwatch import analyze

result = analyze("memories.json")
print(f"Flagged: {len(result['flagged'])}/{result['total_entries']}")
for r in result["flagged"]:
    print(f"  {r.entry.id}: {r.action} (score={r.stale_score})")
```

## License

MIT © Yunare Maia

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Empty scan result | Confirm the path is JSON/JSONL and readable |
| High duplicate score | Deduplicate similar facts before agents write more |
| CI noise | Use `--format json` and fail only on severity thresholds |
