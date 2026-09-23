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

- **YAML** — Single-document or multi-document YAML stores (Mem0, Letta)
- **JSON / JSONL** — `{"facts": [...]}` or line-delimited JSON objects
- **SQLite** — `memories` table with standard columns
- **Auto-detect** — pass any path (`.yaml`, `.yml`, `.json`, `.jsonl`, `.db`), memwatch figures it out

When using the Python API with a custom SQLite table, pass a simple identifier
such as `agent_memories` to `parse_sqlite_store`. Table names are validated
against `[a-zA-Z_][a-zA-Z0-9_]*`; SQL fragments such as `memories; DROP TABLE
memories; --` are rejected. SQLite table names cannot be bound as query
parameters, so do not construct or modify the table name from untrusted input.

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


## Real-world memory profiles

Illustrative patterns you will see when scanning agent memory stores. Numbers are examples — thresholds stay the defaults above unless you change them.

### Healthy profile

Stable agent memory over a long session: most facts score low, few contradictions, no duplicate explosions.

```text
$ memwatch scan memories.json --verbose
Scanned 128 entries
Flagged: 4/128 (3%)

  fact_012: KEEP   score=0.12  age=3d   confirms=4
  fact_044: KEEP   score=0.18  age=9d   confirms=2
  fact_091: REFRESH score=0.62 age=28d  confirms=1  # borderline age
  fact_110: REVIEW  score=0.45 age=5d   contradicts=fact_019

Suggested: reconfirm fact_091; resolve contradiction on fact_110.
```

Signals of health: flagged rate under ~10%, almost no DELETE, confirms growing on core facts.

### Slow leak / rot pattern

Memory grows while quality drops — new duplicates and unresolved contradictions pile up over an hour of agent work.

```text
$ memwatch scan memories.json --format json | head
{
  "total_entries": 640,
  "flagged": 210,
  "actions": {"DELETE": 48, "MERGE": 71, "REVIEW": 55, "REFRESH": 36, "KEEP": 430}
}

$ memwatch scan memories.json --verbose | tail -5
  fact_501: DELETE score=0.91 age=52d confirms=0
  fact_512: MERGE  score=0.70 duplicate_of=fact_088
  fact_520: MERGE  score=0.68 duplicate_of=fact_088
  fact_601: REVIEW score=0.55 contradicts=fact_012
  fact_630: DELETE score=0.88 age=41d confirms=0
```

Signals of rot: flagged rate climbing session over session, many MERGE on the same id, DELETE cluster on never-confirmed facts. Wire `memwatch scan --format json` into CI and fail when `len(flagged)/total_entries` exceeds your budget (e.g. 0.15).

### CI-style gate

```bash
# Example: fail the job when more than 15% of facts are flagged
python - <<'PY'
from memwatch import analyze
r = analyze("memories.json")
rate = len(r["flagged"]) / max(r["total_entries"], 1)
print(f"flagged_rate={rate:.2%}")
raise SystemExit(0 if rate <= 0.15 else 1)
PY
```

## License

MIT © Yunare Maia

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Empty scan result | Confirm the path is JSON/JSONL and readable |
| High duplicate score | Deduplicate similar facts before agents write more |
| CI noise | Use `--format json` and fail only on severity thresholds |
