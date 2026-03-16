"""Generate human-readable reports from ontological fingerprints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from architect.fingerprint import OntologicalFingerprint
from architect.comparator import FingerprintComparator, ComparisonResult, MultiComparisonResult


def print_fingerprint(fp: OntologicalFingerprint, console: Optional[Console] = None) -> None:
    """Pretty-print a single fingerprint to the terminal."""
    con = console or Console()

    con.print(Panel(f"[bold]{fp.model_name}[/bold]", title="Ontological Fingerprint"))

    # Ontology table
    if fp.ontology_scores:
        table = Table(title="Ontological Commitments (existence confidence)")
        table.add_column("Concept", style="cyan")
        table.add_column("Confidence", justify="right")
        for concept, score in sorted(fp.ontology_scores.items()):
            color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
            table.add_row(concept.replace("_", " ").title(), f"[{color}]{score:.2f}[/{color}]")
        con.print(table)

    # Axiology table
    if fp.axiology_scores:
        table = Table(title="Value Hierarchy (average importance)")
        table.add_column("Value", style="cyan")
        table.add_column("Score", justify="right")
        for value, score in sorted(fp.axiology_scores.items(), key=lambda x: x[1], reverse=True):
            table.add_row(value.replace("_", " ").title(), f"{score:.2f}")
        con.print(table)

    # Epistemic table
    if fp.epistemic_scores:
        table = Table(title="Epistemic Confidence")
        table.add_column("Claim", style="cyan")
        table.add_column("Confidence", justify="right")
        for claim, score in sorted(fp.epistemic_scores.items()):
            table.add_row(claim.replace("_", " ").title()[:50], f"{score:.2f}")
        con.print(table)

    # Cultural summary
    if fp.cultural_scores:
        table = Table(title="Cultural Divergence")
        table.add_column("Concept", style="cyan")
        table.add_column("Divergence", justify="right")
        for concept, score in sorted(fp.cultural_scores.items()):
            table.add_row(concept.replace("_", " ").title(), f"{score:.2f}")
        con.print(table)


def print_comparison(result: ComparisonResult, console: Optional[Console] = None) -> None:
    """Pretty-print a pairwise comparison."""
    con = console or Console()

    con.print(Panel(
        f"[bold]{result.model_a}[/bold] vs [bold]{result.model_b}[/bold]",
        title="Fingerprint Comparison",
    ))

    con.print(f"  Cosine similarity:  [bold]{result.cosine_similarity:.4f}[/bold]")
    con.print(f"  Euclidean distance: [bold]{result.euclidean_distance:.4f}[/bold]")

    if result.biggest_differences:
        table = Table(title="Biggest Differences")
        table.add_column("Dimension", style="cyan")
        table.add_column(result.model_a, justify="right")
        table.add_column(result.model_b, justify="right")
        table.add_column("Delta", justify="right")
        for dim, sa, sb in result.biggest_differences[:10]:
            table.add_row(
                dim.replace("_", " ").title(),
                f"{sa:.2f}",
                f"{sb:.2f}",
                f"[bold]{abs(sa - sb):.2f}[/bold]",
            )
        con.print(table)

    if result.shared_blindspots:
        con.print(f"\n  [yellow]Shared blindspots ({len(result.shared_blindspots)}):[/yellow]")
        for bs in result.shared_blindspots[:10]:
            con.print(f"    - {bs.replace('_', ' ').title()}")


def export_fingerprint_json(
    fp: OntologicalFingerprint,
    output_path: str | Path,
) -> Path:
    """Export a fingerprint to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_name": fp.model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ontology_scores": fp.ontology_scores,
        "axiology_scores": fp.axiology_scores,
        "epistemic_scores": fp.epistemic_scores,
        "cultural_scores": fp.cultural_scores,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def generate_full_report(
    fingerprints: Sequence[OntologicalFingerprint],
    *,
    output_dir: Optional[str | Path] = None,
    console: Optional[Console] = None,
) -> None:
    """Generate and display a full report for one or more fingerprints.

    If *output_dir* is provided, also writes JSON exports and chart images.
    """
    con = console or Console()
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    for fp in fingerprints:
        print_fingerprint(fp, console=con)
        if out:
            export_fingerprint_json(fp, out / f"{fp.model_name.replace('/', '_')}.json")

    if len(fingerprints) >= 2:
        comparator = FingerprintComparator()
        multi = comparator.compare_multiple(list(fingerprints))
        for cr in multi.pairwise:
            print_comparison(cr, console=con)

        if multi.consensus_dimensions:
            con.print(f"\n[green]Consensus dimensions ({len(multi.consensus_dimensions)}):[/green]")
            for d in multi.consensus_dimensions[:10]:
                con.print(f"  - {d.replace('_', ' ').title()}")

        if multi.most_divisive_dimensions:
            con.print(f"\n[red]Most divisive dimensions:[/red]")
            for d in multi.most_divisive_dimensions[:10]:
                con.print(f"  - {d.replace('_', ' ').title()}")

        # Generate charts if output dir provided
        if out:
            from architect.visualizer import radar_chart, heatmap, value_comparison_bar

            try:
                radar_chart(fingerprints, output_path=out / "radar_ontology.png")
                heatmap(fingerprints, score_type="ontology", output_path=out / "heatmap_ontology.png")
                heatmap(fingerprints, score_type="epistemic", title="Epistemic Confidence Heatmap",
                        output_path=out / "heatmap_epistemic.png")
                value_comparison_bar(fingerprints, output_path=out / "bar_axiology.png")
                con.print(f"\n[green]Charts saved to {out}/[/green]")
            except Exception as exc:
                con.print(f"\n[yellow]Chart generation failed: {exc}[/yellow]")
