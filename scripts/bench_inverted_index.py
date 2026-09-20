#!/usr/bin/env python3
"""Benchmark: O(n²) scan vs inverted index for contradiction detection."""
import time
from datetime import datetime, timezone
from memwatch.core import MemoryEntry, ContradictionIndex, _detect_contradictions_scan


def make_entries(n: int) -> list[MemoryEntry]:
    now = datetime.now(timezone.utc)
    return [
        MemoryEntry(id=f"e{i}", content=f"Uses library{i % 100} for auth and database", created_at=now)
        for i in range(n)
    ]


def benchmark(entries: list[MemoryEntry]) -> None:
    n = len(entries)
    start = time.perf_counter()
    _detect_contradictions_scan(entries)
    scan_time = time.perf_counter() - start

    start = time.perf_counter()
    idx = ContradictionIndex(entries)
    idx.find_contradictions()
    index_time = time.perf_counter() - start

    speedup = scan_time / index_time if index_time > 0 else float("inf")
    print(f"n={n:>5d}: scan={scan_time:.3f}s  index={index_time:.3f}s  speedup={speedup:.1f}x")


if __name__ == "__main__":
    for n in [100, 500, 1000, 2000]:
        benchmark(make_entries(n))
