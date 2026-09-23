"""Edge-case tests for :mod:`memwatch.core`.

The existing ``test_core.py`` covers the happy paths of the scoring, contradiction
and deduplication helpers. This module covers the gaps that matter in practice —
the ones the scan is most likely to hit with real agent memory stores:

* **empty stores** — a valid file that holds no entries
* **malformed entries** — missing fields, wrong types, empty strings
* **None values** — fields explicitly set to ``null`` in the source data

It also covers several public helpers that had no tests at all before:
``_parse_ts``, ``parse_store``, ``_normalize``, ``compute_stale_score`` and the
derived ``StaleReport.action`` / ``needs_attention`` properties.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from memwatch.core import (
    MemoryEntry,
    SAFE_TABLE_RE,
    StaleReport,
    _age_score,
    _are_contradictory,
    _confirmation_score,
    _jaccard,
    _normalize,
    _parse_ts,
    analyze,
    compute_stale_score,
    find_duplicates,
    parse_json_store,
    parse_sqlite_store,
    parse_store,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

def _write_json(tmp_path, payload, name="store.json"):
    """Write ``payload`` as JSON and return its path as a string."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _make_db(tmp_path, rows, table="memories", name="memories.db"):
    """Create a SQLite store with the standard columns and return its path."""
    db_path = tmp_path / name
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        f"CREATE TABLE {table} ("
        "id TEXT PRIMARY KEY, content TEXT, created_at TEXT, "
        "confirmed_count INTEGER DEFAULT 0)"
    )
    conn.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return str(db_path)


def _entry(entry_id="e1", content="some fact", days_old=0, confirmed=0):
    """Build a MemoryEntry positioned ``days_old`` days in the past."""
    return MemoryEntry(
        id=entry_id,
        content=content,
        created_at=datetime.now(timezone.utc) - timedelta(days=days_old),
        confirmed_count=confirmed,
    )


# ── Empty stores ─────────────────────────────────────────────────────────

class TestEmptyStores:
    """A valid store that happens to hold no entries must not blow up."""

    def test_empty_json_list_parses_to_no_entries(self, tmp_path):
        path = _write_json(tmp_path, [])
        assert parse_json_store(path) == []

    @pytest.mark.xfail(
        reason="known bug: an empty `facts` list falls through to the "
               "'treat top-level keys as entries' branch, so {\"facts\": []} "
               "parses to one bogus entry (id='facts', content='[]')",
        strict=True,
    )
    def test_empty_facts_key_parses_to_no_entries(self, tmp_path):
        """``{"facts": []}`` is an empty store, not a one-entry store."""
        path = _write_json(tmp_path, {"facts": []})
        assert parse_json_store(path) == []

    def test_empty_sqlite_table_parses_to_no_entries(self, tmp_path):
        path = _make_db(tmp_path, [])
        assert parse_sqlite_store(path) == []

    def test_analyze_empty_store_reports_zero(self, tmp_path):
        path = _write_json(tmp_path, [])
        result = analyze(path)
        assert result["total_entries"] == 0
        assert result["flagged"] == []
        assert result["healthy"] == []
        # avg_stale divides by max(len, 1) so an empty store gives 0.0
        assert result["stats"]["avg_stale"] == 0.0

    def test_analyze_empty_store_keeps_stats_keys(self, tmp_path):
        result = analyze(_write_json(tmp_path, []))
        assert set(result["stats"]) == {
            "avg_stale", "high_risk", "with_contradictions", "with_duplicates",
        }


# ── Malformed entries ────────────────────────────────────────────────────

class TestMalformedEntries:
    """Entries missing fields or carrying the wrong types must be tolerated."""

    def test_entry_missing_everything_but_id(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": "bare"}]})
        entry = parse_json_store(path)[0]
        assert entry.id == "bare"
        assert entry.content == ""
        assert entry.confirmed_count == 0
        # An unparseable timestamp falls back to "now", so the entry is fresh.
        assert entry.age_days < 1

    def test_entry_without_id_gets_one(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"content": "no id here"}]})
        entry = parse_json_store(path)[0]
        assert entry.id            # generated, not empty
        assert entry.content == "no id here"

    def test_string_items_become_entries(self, tmp_path):
        path = _write_json(tmp_path, {"facts": ["plain string fact"]})
        entry = parse_json_store(path)[0]
        assert entry.id == "plain string fact"
        assert entry.content == "plain string fact"

    def test_alternate_field_names_are_accepted(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [
            {"key": "k1", "text": "uses text field", "timestamp": 1700000000,
             "confirmations": 3},
        ]})
        entry = parse_json_store(path)[0]
        assert entry.id == "k1"
        assert entry.content == "uses text field"
        assert entry.confirmed_count == 3

    def test_numeric_id_is_coerced_to_string(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": 42, "content": "x"}]})
        assert parse_json_store(path)[0].id == "42"

    def test_confirmed_count_given_as_numeric_string(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": "a", "content": "x",
                                                 "confirmed_count": "7"}]})
        assert parse_json_store(path)[0].confirmed_count == 7

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{ not json at all", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            parse_json_store(str(path))

    def test_unrecognised_top_level_keys_become_entries(self, tmp_path):
        """A dict whose keys are not facts/episodes/... is read as id -> value."""
        path = _write_json(tmp_path, {"alpha": "first value", "beta": "second"})
        ids = sorted(e.id for e in parse_json_store(path))
        assert ids == ["alpha", "beta"]

    def test_invalid_timestamp_values_are_skipped(self, tmp_path, caplog):
        path = _write_json(tmp_path, {"facts": [
            {"id": "a", "content": "x", "created_at": "not-a-date"},
            {"id": "b", "content": "y", "created_at": 12345},
            {"id": "c", "content": "z", "created_at": []},
        ]})
        entries = parse_json_store(path)
        assert [entry.id for entry in entries] == ["b"]
        assert isinstance(entries[0].created_at, datetime)
        assert "entry 1" in caplog.text
        assert "entry 3" in caplog.text

    def test_invalid_confirmed_count_is_skipped(self, tmp_path, caplog):
        path = _write_json(tmp_path, {"facts": [
            {"id": "bad", "content": "x", "confirmed_count": "many"},
            {"id": "good", "content": "y", "confirmed_count": 2},
        ]})

        entries = parse_json_store(path)

        assert [entry.id for entry in entries] == ["good"]
        assert "entry 1" in caplog.text
        assert "confirmed_count is not a valid integer" in caplog.text

    def test_analyze_summarizes_skipped_entries(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [
            {"id": "bad-content", "content": None},
            {"id": "bad-count", "content": "x", "confirmed_count": []},
            {"id": "good", "content": "y"},
        ]})

        result = analyze(path)

        assert result["total_entries"] == 1
        assert result["skipped_entries"] == 2
        assert len(result["skipped_entry_details"]) == 2


# ── None values ──────────────────────────────────────────────────────────

class TestNoneValues:
    """Explicit ``null`` values are handled according to each field's schema."""

    def test_none_content_is_skipped(self, tmp_path, caplog):
        path = _write_json(tmp_path, {"facts": [{"id": "a", "content": None}]})
        assert parse_json_store(path) == []
        assert "content is not a string" in caplog.text

    def test_none_created_at_falls_back_to_now(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": "a", "content": "x",
                                                 "created_at": None}]})
        entry = parse_json_store(path)[0]
        assert isinstance(entry.created_at, datetime)
        assert entry.age_days < 1

    def test_none_last_confirmed_at_stays_none(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": "a", "content": "x",
                                                 "last_confirmed_at": None}]})
        assert parse_json_store(path)[0].last_confirmed_at is None

    def test_parse_ts_handles_none(self):
        assert isinstance(_parse_ts(None), datetime)

    def test_parse_ts_handles_empty_string(self):
        assert isinstance(_parse_ts(""), datetime)

    def test_parse_ts_handles_zero(self):
        """0 is falsy, so it takes the "now" branch rather than 1970."""
        assert isinstance(_parse_ts(0), datetime)

    @pytest.mark.xfail(
        reason="known bug: parse_sqlite_store calls int() on confirmed_count "
               "without a None guard, so a NULL column raises TypeError",
        strict=True,
    )
    def test_sqlite_null_columns_do_not_crash(self, tmp_path):
        path = _make_db(tmp_path, [("a", None, None, None)])
        entry = parse_sqlite_store(path)[0]
        assert entry.id == "a"
        assert isinstance(entry.created_at, datetime)


# ── Timestamp parsing ────────────────────────────────────────────────────

class TestTimestampParsing:
    """``_parse_ts`` is the single funnel for every timestamp in the module."""

    def test_aware_datetime_is_returned_unchanged(self):
        original = datetime(2024, 5, 1, tzinfo=timezone.utc)
        assert _parse_ts(original) == original

    def test_naive_datetime_gets_utc(self):
        naive = datetime(2024, 5, 1, 12, 0, 0)
        parsed = _parse_ts(naive)
        assert parsed.tzinfo is not None
        assert parsed == naive.replace(tzinfo=timezone.utc)

    def test_iso_string_with_z_suffix(self):
        parsed = _parse_ts("2024-05-01T12:00:00Z")
        assert parsed.year == 2024 and parsed.month == 5 and parsed.day == 1
        assert parsed.tzinfo is not None

    def test_iso_string_with_offset(self):
        parsed = _parse_ts("2024-05-01T12:00:00+08:00")
        assert parsed.tzinfo is not None

    def test_iso_string_without_timezone(self):
        parsed = _parse_ts("2024-05-01T12:00:00")
        assert parsed.tzinfo is not None

    def test_epoch_seconds(self):
        assert _parse_ts(1700000000).year == 2023

    def test_epoch_float(self):
        assert isinstance(_parse_ts(1700000000.5), datetime)

    def test_garbage_string_falls_back_to_now(self):
        assert _parse_ts("yesterday").year >= 2026

    def test_unparseable_returns_now_not_raise(self):
        """A bad timestamp must never make the whole scan fail."""
        for bad in ([], {}, "!!!"):
            assert isinstance(_parse_ts(bad), datetime)


# ── Store autodetection ──────────────────────────────────────────────────

class TestParseStoreDispatch:
    """``parse_store`` picks a parser from the file extension."""

    def test_json_extension(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": "a", "content": "x"}]})
        assert len(parse_store(path)) == 1

    def test_db_extension(self, tmp_path):
        path = _make_db(tmp_path, [("a", "x", "2026-01-01T00:00:00Z", 0)])
        assert len(parse_store(path)) == 1

    def test_sqlite_extension(self, tmp_path):
        path = _make_db(tmp_path, [("a", "x", "2026-01-01T00:00:00Z", 0)],
                        name="memories.sqlite")
        assert len(parse_store(path)) == 1

    def test_unknown_extension_falls_back_to_sqlite(self, tmp_path):
        """With no recognised suffix the JSON attempt fails, then SQLite runs."""
        path = _make_db(tmp_path, [("a", "x", "2026-01-01T00:00:00Z", 0)],
                        name="memories.data")
        assert len(parse_store(path)) == 1


# ── Text normalisation ───────────────────────────────────────────────────

class TestNormalize:
    """``_normalize`` feeds both contradiction and duplication checks."""

    def test_expands_contraction(self):
        assert "not" in _normalize("doesn't use redis")

    def test_lowercases(self):
        assert _normalize("REDIS") == _normalize("redis")

    def test_strips_punctuation(self):
        assert _normalize("uses redis.") == _normalize("uses redis")

    def test_singularises_long_words(self):
        """Trailing "s" is dropped for words longer than three characters."""
        assert "database" in _normalize("databases")

    def test_strips_trailing_s_from_words_longer_than_three(self):
        """Singular and plural forms collapse onto the same token."""
        assert _normalize("runs") == _normalize("run")
        assert _normalize("databases") == _normalize("database")

    def test_words_of_three_characters_or_fewer_are_untouched(self):
        assert _normalize("is") == {"is"}
        assert _normalize("as") == {"as"}

    def test_empty_text_gives_empty_set(self):
        assert _normalize("") == set()

    def test_punctuation_only_gives_empty_set(self):
        assert _normalize("...!!!") == set()


class TestConservativeStemming:
    """Issue #63: stemming must not mangle non-plural ``s`` endings."""

    def test_tests_and_test_collapse_to_one_token(self):
        assert _normalize("tests") == _normalize("test") == {"test"}

    def test_uses_and_user_do_not_collide(self):
        # "uses" stems to "use"; it must not become "user".
        assert _normalize("uses") != _normalize("user")

    def test_address_is_not_modified(self):
        assert _normalize("address") == {"address"}

    def test_process_is_not_modified(self):
        assert _normalize("process") == {"process"}

    def test_words_ending_in_us_are_not_modified(self):
        assert _normalize("bonus") == {"bonus"}
        assert _normalize("status") == {"status"}

    def test_words_ending_in_is_are_not_modified(self):
        assert _normalize("iris") == {"iris"}
        assert _normalize("analysis") == {"analysis"}

    def test_genuine_plurals_still_collapse(self):
        assert _normalize("sessions") == _normalize("session") == {"session"}

    def test_self_match_no_false_contradiction(self):
        """'address' no longer stems to 'addres' and fails to match itself."""
        from memwatch.core import _are_contradictory

        # Same shared words including "address"; not contradictory (no negation flip).
        assert not _are_contradictory(
            "the service address is fixed and stable",
            "the service address is changed and unstable",
        )


# ── Stale scoring details ────────────────────────────────────────────────

class TestStaleScoreReasons:
    """``compute_stale_score`` attaches human-readable reasons."""

    def test_fresh_confirmed_entry_has_no_reasons(self):
        report = compute_stale_score(_entry(days_old=1, confirmed=5), [])
        assert report.reasons == []

    def test_old_entry_is_flagged_as_old(self):
        report = compute_stale_score(_entry(days_old=100, confirmed=5), [])
        assert any("Old" in r for r in report.reasons)

    def test_unconfirmed_entry_is_flagged(self):
        report = compute_stale_score(_entry(days_old=1, confirmed=0), [])
        assert "Never confirmed" in report.reasons

    def test_stale_and_unconfirmed_pair(self):
        report = compute_stale_score(_entry(days_old=40, confirmed=0), [])
        assert "Stale + unconfirmed" in report.reasons

    def test_recent_unconfirmed_is_not_called_stale(self):
        """Under 30 days the "Stale + unconfirmed" reason does not apply."""
        report = compute_stale_score(_entry(days_old=5, confirmed=0), [])
        assert "Stale + unconfirmed" not in report.reasons

    def test_score_is_rounded_to_three_places(self):
        report = compute_stale_score(_entry(days_old=17, confirmed=2), [])
        assert report.stale_score == round(report.stale_score, 3)

    def test_score_never_exceeds_one(self):
        report = compute_stale_score(_entry(days_old=5000, confirmed=0), [])
        assert 0.0 <= report.stale_score <= 1.0


class TestScoreHelpersAtBoundaries:
    def test_age_score_of_brand_new_entry_is_near_zero(self):
        assert _age_score(_entry(days_old=0)) == pytest.approx(0.0, abs=0.01)

    def test_age_score_at_half_life_is_one_half(self):
        assert _age_score(_entry(days_old=30)) == pytest.approx(0.5, abs=0.02)

    def test_age_score_approaches_one_for_ancient_entries(self):
        assert _age_score(_entry(days_old=3650)) > 0.99

    def test_confirmation_score_is_never_negative(self):
        assert _confirmation_score(_entry(confirmed=1000)) == 0.0

    def test_confirmation_score_drops_with_each_confirmation(self):
        scores = [_confirmation_score(_entry(confirmed=n)) for n in range(6)]
        assert scores == sorted(scores, reverse=True)


# ── StaleReport derived properties ───────────────────────────────────────

class TestStaleReportProperties:
    """``action`` is an if/elif chain, so order decides the outcome."""

    def test_delete_wins_over_duplicates(self):
        """score > 0.8 is checked first, so a very stale duplicate is DELETE."""
        report = StaleReport(entry=_entry(), stale_score=0.95, reasons=[],
                             duplicates=[_entry("other")])
        assert report.action == "DELETE"

    def test_contradiction_gives_review(self):
        report = StaleReport(entry=_entry(), stale_score=0.5, reasons=[],
                             contradictions=[_entry("other")])
        assert report.action == "REVIEW"

    def test_duplicate_gives_merge(self):
        report = StaleReport(entry=_entry(), stale_score=0.7, reasons=[],
                             duplicates=[_entry("other")])
        assert report.action == "MERGE"

    def test_moderately_stale_gives_refresh(self):
        report = StaleReport(entry=_entry(), stale_score=0.65, reasons=[])
        assert report.action == "REFRESH"

    def test_fresh_gives_keep(self):
        report = StaleReport(entry=_entry(), stale_score=0.1, reasons=[])
        assert report.action == "KEEP"

    def test_action_at_exact_boundaries(self):
        """The comparisons are strict, so 0.6 and 0.8 fall to the lower action."""
        assert StaleReport(entry=_entry(), stale_score=0.6, reasons=[]).action == "KEEP"
        assert StaleReport(entry=_entry(), stale_score=0.8, reasons=[]).action == "REFRESH"

    def test_needs_attention_on_high_score(self):
        assert StaleReport(entry=_entry(), stale_score=0.61, reasons=[]).needs_attention

    def test_needs_attention_on_contradiction_even_when_fresh(self):
        report = StaleReport(entry=_entry(), stale_score=0.05, reasons=[],
                             contradictions=[_entry("other")])
        assert report.needs_attention

    def test_duplicates_alone_do_not_trigger_attention(self):
        """A duplicate with a low score is not "flagged" by needs_attention."""
        report = StaleReport(entry=_entry(), stale_score=0.1, reasons=[],
                             duplicates=[_entry("other")])
        assert not report.needs_attention


# ── Contradiction and duplication boundaries ─────────────────────────────

class TestContradictionBoundaries:
    @pytest.mark.xfail(
        reason="known bug: the final 'not X' fallback returns True on a single "
               "shared word, bypassing the two-word overlap threshold used by "
               "the NEGATION_PAIRS branch",
        strict=True,
    )
    def test_single_shared_word_is_not_enough(self):
        """The overlap threshold should also apply to the 'not X' fallback.

        ``redis is fast`` and ``memcached is not fast`` share exactly one
        meaningful word, so this should not be reported as a contradiction.
        """
        assert not _are_contradictory("redis is fast", "memcached is not fast")

    def test_empty_strings_are_not_contradictory(self):
        assert not _are_contradictory("", "")

    def test_identical_text_is_not_contradictory(self):
        assert not _are_contradictory("uses redis for caching",
                                      "uses redis for caching")

    def test_case_and_punctuation_do_not_matter(self):
        assert _are_contradictory("Uses Redis for caching!",
                                  "doesn't use redis for caching")

    def test_negation_pair_with_shared_subject(self):
        assert _are_contradictory("feature is enabled", "feature is disabled")


class TestDuplicateBoundaries:
    def test_empty_content_never_duplicates(self):
        assert _jaccard("", "") == 0.0

    def test_one_empty_side_gives_zero(self):
        assert _jaccard("uses redis", "") == 0.0

    def test_identical_content_is_one(self):
        assert _jaccard("uses redis for caching", "uses redis for caching") == 1.0

    def test_find_duplicates_on_empty_list(self):
        assert find_duplicates([]) == {}

    def test_find_duplicates_ignores_unique_entries(self):
        entries = [_entry("a", "uses redis for caching"),
                   _entry("b", "deploys with docker compose")]
        assert find_duplicates(entries) == {}

    def test_threshold_is_respected(self):
        """A low threshold groups near-identical text; a high one does not."""
        entries = [_entry("a", "uses redis for caching"),
                   _entry("b", "uses redis for storing")]
        assert find_duplicates(entries, threshold=0.5) != {}
        assert find_duplicates(entries, threshold=0.99) == {}


# ── End-to-end analysis on awkward input ─────────────────────────────────

class TestAnalyzeEdgeCases:
    def test_store_with_one_entry(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [{"id": "only", "content": "x"}]})
        result = analyze(path)
        assert result["total_entries"] == 1
        assert len(result["reports"]) == 1

    def test_empty_content_entries_do_not_crash(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [
            {"id": "a", "content": ""},
            {"id": "b", "content": ""},
        ]})
        result = analyze(path)
        assert result["total_entries"] == 2

    def test_flagged_and_healthy_partition_every_entry(self, tmp_path):
        path = _write_json(tmp_path, {"facts": [
            {"id": "old", "content": "ancient fact", "created_at": "2000-01-01T00:00:00Z"},
            {"id": "new", "content": "fresh fact",
             "created_at": datetime.now(timezone.utc).isoformat(), "confirmed_count": 5},
        ]})
        result = analyze(path)
        assert len(result["flagged"]) + len(result["healthy"]) == result["total_entries"]

    def test_sqlite_store_end_to_end(self, tmp_path):
        path = _make_db(tmp_path, [
            ("a", "uses redis for caching", "2000-01-01T00:00:00Z", 0),
            ("b", "deploys with docker", "2026-01-01T00:00:00Z", 5),
        ])
        result = analyze(path)
        assert result["total_entries"] == 2


# ── Guard rail for the SQL-name validation ───────────────────────────────

class TestTableNameValidation:
    """``parse_sqlite_store`` validates the table name before interpolating it."""

    @pytest.mark.parametrize("bad", [
        "memories; DROP TABLE memories",
        "memories--",
        "1memories",
        "mem ories",
        "",
        "memories'",
    ])
    def test_unsafe_table_names_are_rejected(self, tmp_path, bad):
        path = _make_db(tmp_path, [])
        with pytest.raises(ValueError):
            parse_sqlite_store(path, table=bad)

    @pytest.mark.parametrize("good", ["memories", "_memories", "memories_2", "M2"])
    def test_safe_table_names_match_the_pattern(self, good):
        assert SAFE_TABLE_RE.match(good)

    def test_custom_table_name_is_usable(self, tmp_path):
        path = _make_db(tmp_path, [("a", "x", "2026-01-01T00:00:00Z", 0)],
                        table="custom_table")
        assert len(parse_sqlite_store(path, table="custom_table")) == 1
