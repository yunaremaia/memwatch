# Changelog

All notable changes to memwatch will be documented in this file.

## [Unreleased]

### Added
- Inverted index for O(n) contradiction detection (issue #53). Auto-enabled for stores >500 entries.
- `ContradictionIndex` class with token-based candidate lookup.
- `--index/--no-index` CLI flag to force-enable/disable the inverted index.
- JSON and JSONL schema validation with skipped-entry details in analysis results.
