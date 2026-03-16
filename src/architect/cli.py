"""CLI entry point for Architect."""

from __future__ import annotations

import click
from rich.console import Console

from architect.models import get_model

console = Console()


@click.group()
@click.version_option(package_name="architect-ai")
def cli() -> None:
    """Architect -- map the ontological fingerprint of AI models."""


@cli.command()
@click.option("--model", "-m", required=True, help="Model name (e.g. claude, gpt-4o)")
@click.option(
    "--type", "-t", "probe_type",
    type=click.Choice(["ontological", "axiological", "epistemic", "cultural"]),
    required=True,
    help="Probe type to run",
)
@click.option("--output", "-o", default=None, help="Output JSON path")
def probe(model: str, probe_type: str, output: str | None) -> None:
    """Run a single probe battery against a model."""
    import json

    adapter = get_model(model)
    console.print(f"Running [bold]{probe_type}[/bold] probe against [cyan]{adapter.name}[/cyan]...")

    if probe_type == "ontological":
        from architect.probes.ontological import OntologicalProbe
        results = OntologicalProbe().run(adapter)
        data = [
            {"concept": r.concept, "existence_confidence": r.existence_confidence,
             "reasoning": r.reasoning, "category": r.category}
            for r in results
        ]
    elif probe_type == "axiological":
        from architect.probes.axiological import AxiologicalProbe
        results = AxiologicalProbe().run(adapter)
        data = [
            {"framing": r.framing, "rankings": r.rankings,
             "reasoning": r.reasoning, "category": r.category}
            for r in results
        ]
    elif probe_type == "epistemic":
        from architect.probes.epistemic import EpistemicProbe
        results = EpistemicProbe().run(adapter)
        data = [
            {"claim": r.claim, "stated_confidence": r.stated_confidence,
             "reasoning": r.reasoning, "category": r.category}
            for r in results
        ]
    elif probe_type == "cultural":
        from architect.probes.cultural import CulturalProbe
        results = CulturalProbe().run(adapter)
        data = [
            {"concept": r.concept, "divergence_score": r.divergence_score,
             "perspectives": [
                 {"culture": p.culture, "key_themes": p.key_themes, "emphasis": p.emphasis}
                 for p in r.perspectives
             ]}
            for r in results
        ]
    else:
        raise click.BadParameter(f"Unknown probe type: {probe_type}")

    if output:
        from pathlib import Path
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Results saved to {output}[/green]")
    else:
        console.print_json(json.dumps(data, indent=2))


@cli.command()
@click.option("--model", "-m", required=True, help="Model name (e.g. claude, gpt-4o)")
@click.option("--output", "-o", default=None, help="Output directory for reports/charts")
@click.option("--skip-cultural", is_flag=True, help="Skip cultural probes (saves API calls)")
def fingerprint(model: str, output: str | None, skip_cultural: bool) -> None:
    """Generate a full ontological fingerprint for a model."""
    from architect.fingerprint import generate_fingerprint
    from architect.report import print_fingerprint, export_fingerprint_json

    adapter = get_model(model)
    console.print(f"Generating fingerprint for [cyan]{adapter.name}[/cyan]...")

    fp = generate_fingerprint(adapter, skip_cultural=skip_cultural)
    print_fingerprint(fp, console=console)

    if output:
        from pathlib import Path
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = export_fingerprint_json(fp, out_dir / f"{fp.model_name.replace('/', '_')}.json")
        console.print(f"[green]Fingerprint saved to {path}[/green]")


@cli.command()
@click.option(
    "--models", "-m", required=True,
    help="Comma-separated model names (e.g. claude,gpt-4o)",
)
@click.option("--output", "-o", default=None, help="Output directory for reports/charts")
@click.option("--skip-cultural", is_flag=True, help="Skip cultural probes (saves API calls)")
def compare(models: str, output: str | None, skip_cultural: bool) -> None:
    """Compare ontological fingerprints across multiple models."""
    from architect.fingerprint import generate_fingerprint
    from architect.report import generate_full_report

    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_names) < 2:
        raise click.BadParameter("Provide at least 2 model names separated by commas.")

    fingerprints = []
    for name in model_names:
        adapter = get_model(name)
        console.print(f"Generating fingerprint for [cyan]{adapter.name}[/cyan]...")
        fp = generate_fingerprint(adapter, skip_cultural=skip_cultural)
        fingerprints.append(fp)

    generate_full_report(fingerprints, output_dir=output, console=console)


if __name__ == "__main__":
    cli()
