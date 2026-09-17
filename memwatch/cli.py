"""CLI for memwatch — agent memory health monitor."""

import json
import sys
from pathlib import Path

import click
from memwatch.core import analyze, StaleReport


def _format_report(report: StaleReport, verbose: bool = False) -> str:
    """Format a single report for terminal output."""
    entry = report.entry
    action_color = {
        "DELETE": "red",
        "REVIEW": "yellow",
        "MERGE": "cyan",
        "REFRESH": "blue",
        "KEEP": "green",
    }.get(report.action, "white")

    lines = [
        f"[{report.action}] {entry.id[:40]} (score={report.stale_score:.2f}, age={entry.age_days:.0f}d)",
    ]
    if report.reasons:
        lines.append(f"  Reasons: {'; '.join(report.reasons)}")
    if verbose:
        lines.append(f"  Content: {entry.content[:120]}")
    if report.contradictions:
        lines.append(f"  ⚠ {len(report.contradictions)} contradiction(s)")
    if report.duplicates:
        lines.append(f"  ⓘ {len(report.duplicates)} duplicate(s)")
    return "\n".join(lines)


@click.group()
@click.version_option(package_name="memwatch")
def cli():
    """memwatch — monitor and fix AI agent memory rot."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--threshold", default=0.6, help="Stale score threshold (0-1)")
@click.option("--verbose", "-v", is_flag=True, help="Show full content")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def scan(path, threshold, verbose, fmt):
    """Scan a memory store and report health."""
    result = analyze(path, stale_threshold=threshold)

    if fmt == "json":
        output = {
            "store": str(path),
            "stats": result["stats"],
            "flagged": [
                {
                    "id": r.entry.id,
                    "score": r.stale_score,
                    "action": r.action,
                    "reasons": r.reasons,
                    "duplicates": [d.id for d in r.duplicates],
                    "contradictions": [c.id for c in r.contradictions],
                }
                for r in result["flagged"]
            ],
        }
        click.echo(json.dumps(output, indent=2))
        return

    # Text format
    click.echo(f"📊 memwatch — scanning {path}\n")
    stats = result["stats"]
    click.echo(f"Entries: {result['total_entries']} | Avg stale: {stats['avg_stale']:.2f}")
    click.echo(f"High risk: {stats['high_risk']} | Contradictions: {stats['with_contradictions']} | Duplicates: {stats['with_duplicates']}")

    if result["flagged"]:
        click.echo(f"\n🚨 {len(result['flagged'])} entries need attention:\n")
        for r in sorted(result["flagged"], key=lambda x: -x.stale_score):
            click.echo(_format_report(r, verbose))
    else:
        click.echo("\n✅ Memory store is healthy!")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.option("--threshold", default=0.8, help="Delete threshold (0-1)")
def fix(path, dry_run, threshold):
    """Suggest or apply fixes for stale entries."""
    result = analyze(path, stale_threshold=threshold)
    to_delete = [r for r in result["reports"] if r.action == "DELETE"]
    to_merge = [r for r in result["reports"] if r.action == "MERGE"]

    if not to_delete and not to_merge:
        click.echo("✅ Nothing to fix!")
        return

    click.echo(f"🔧 Fix suggestions for {path}:\n")

    if to_delete:
        click.echo(f"DELETE ({len(to_delete)} entries with score > {threshold}):")
        for r in to_delete:
            click.echo(f"  - {r.entry.id[:50]} (score={r.stale_score:.2f})")
        if not dry_run:
            if click.confirm(f"\nDelete {len(to_delete)} entries?"):
                click.echo("Deleted (placeholder — full implementation writes back to store)")

    if to_merge:
        click.echo(f"\nMERGE ({len(to_merge)} entries with duplicates):")
        for r in to_merge:
            ids = [d.id[:30] for d in r.duplicates]
            click.echo(f"  - {r.entry.id[:50]} ↔ {', '.join(ids)}")
        if not dry_run:
            click.echo("(Merge requires manual review — showing suggestions only)")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def dashboard(path, fmt):
    """Show health dashboard (text) or JSON export for tools/CI."""
    result = analyze(path)
    stats = result["stats"]
    total = result["total_entries"]

    if fmt == "json":
        click.echo(json.dumps({
            "store": str(path),
            "total_entries": total,
            "stats": stats,
            "flagged": [
                {
                    "id": r.entry.id,
                    "score": r.stale_score,
                    "action": r.action,
                    "reasons": r.reasons,
                }
                for r in result.get("flagged", result.get("reports", []))
            ],
        }, indent=2))
        return

    click.echo("┌─────────────────────────────────────┐")
    click.echo("│       📊 MEMWATCH DASHBOARD         │")
    click.echo("├─────────────────────────────────────┤")
    click.echo(f"│ Store: {path[:28]:<28} │")
    click.echo(f"│ Entries: {total:<27} │")
    click.echo(f"│ Avg stale: {stats['avg_stale']:.2f}/1.00{'':<22} │")
    click.echo(f"│ High risk: {stats['high_risk']:<26} │")
    click.echo(f"│ Contradictions: {stats['with_contradictions']:<21} │")
    click.echo(f"│ Duplicates: {stats['with_duplicates']:<24} │")
    click.echo("└─────────────────────────────────────┘")


@cli.command("report")
@click.argument("path", type=click.Path(exists=True))
@click.option("--threshold", default=0.6, help="Stale score threshold (0-1)")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="json")
def report(path, threshold, fmt):
    """Export a health report (default JSON for CI/tools)."""
    # Reuse scan behavior by invoking the same analyze path.
    ctx = click.get_current_context()
    ctx.invoke(scan, path=path, threshold=threshold, verbose=False, fmt=fmt)


def main():
    cli()


if __name__ == "__main__":
    main()
