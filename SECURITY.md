# Security Policy

## Scope

`memwatch` reads and rewrites AI agent memory stores (JSON, JSONL, YAML, and
SQLite files) on the machine where it runs. It is a local CLI tool: it makes
no network calls, sends no telemetry, and has no server component. The
security-sensitive surface is the store file itself and the `fix` command that
rewrites it.

## Reporting a vulnerability

The preferred channel is a private
[security advisory](https://github.com/yunaremaia/memwatch/security/advisories/new)
on this repository. If advisories are not enabled, open a regular issue and
avoid posting exploit details — describe the impact and how to reproduce, and
a maintainer will follow up.

Please include:

- what you ran (command, flags, store format),
- what you expected versus what happened,
- a minimal store file that reproduces it, with any secrets redacted.

There is no paid bounty. Expect an acknowledgement within a week and a fix or
a workaround proposal within a month for confirmed issues. Past fixes are
tracked like any other change (for example #70 fixed a query-injection issue
reported as #69).

## If you run memwatch on sensitive data

Memory stores can contain credentials, tokens, and private context the agent
picked up. Treat the store file accordingly:

- **File permissions.** The SQLite backend opens whatever path you pass it.
  Keep store files readable only by you (`chmod 600`), especially on shared
  machines. `memwatch` never relaxes permissions itself, but it does not
  tighten them either.
- **Do not paste raw stores into public issues.** Redact secrets before
  sharing a reproducer, for bug reports as well as security reports.
- **Table names are validated, not quoted.** When using the Python API with a
  custom SQLite table, `parse_sqlite_store` only accepts simple identifiers
  (`[a-zA-Z_][a-zA-Z0-9_]*`). Do not build table names from untrusted input;
  they cannot be bound as query parameters.

## Safety guarantees of `fix`

- `--dry-run` prints what would be deleted and changes nothing. Prefer it for
  a first pass over a store you care about.
- Without `--dry-run`, `fix` asks for confirmation before deleting, but the
  write-back is currently a placeholder — nothing is rewritten yet. Keep a
  copy of stores you care about anyway (`cp memories.json memories.json.bak`),
  since a future version will write back in place and there is no undo.
- `scan` and `dashboard` are read-only and never modify the store.

## Related work

Core hardening is tracked in #73 (SQLite schema validation), #74 (CLI
remediation), and #75 (bootstrap). Security follow-ups belong with those
issues or in a new issue referencing them.
