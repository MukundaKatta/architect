"""Visualization utilities for ontological fingerprints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from architect.fingerprint import OntologicalFingerprint


def radar_chart(
    fingerprints: Sequence[OntologicalFingerprint],
    *,
    title: str = "Ontological Fingerprint Comparison",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Create a radar chart comparing ontological dimensions across models.

    Parameters
    ----------
    fingerprints:
        One or more fingerprints to overlay on the same radar chart.
    title:
        Chart title.
    output_path:
        If provided, save the figure to this path.
    """
    # Collect the union of ontology dimension keys
    all_keys: list[str] = []
    seen: set[str] = set()
    for fp in fingerprints:
        for k in fp.ontology_scores:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if not all_keys:
        raise ValueError("No ontology scores to plot.")

    n = len(all_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(fingerprints), 1)))

    for idx, fp in enumerate(fingerprints):
        values = [fp.ontology_scores.get(k, 0.5) for k in all_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=fp.model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace("_", " ").title() for k in all_keys], size=8)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig


def heatmap(
    fingerprints: Sequence[OntologicalFingerprint],
    *,
    score_type: str = "ontology",
    title: str = "Existence Beliefs Heatmap",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Create a heatmap of scores across models and dimensions.

    Parameters
    ----------
    fingerprints:
        One or more fingerprints.
    score_type:
        Which score dict to use: ``"ontology"``, ``"axiology"``, ``"epistemic"``,
        or ``"cultural"``.
    title:
        Chart title.
    output_path:
        If provided, save the figure to this path.
    """
    score_attr = f"{score_type}_scores"

    # Collect all dimension keys
    all_keys: list[str] = []
    seen: set[str] = set()
    for fp in fingerprints:
        for k in getattr(fp, score_attr, {}):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if not all_keys:
        raise ValueError(f"No {score_type} scores to plot.")

    model_names = [fp.model_name for fp in fingerprints]
    data = np.array(
        [
            [getattr(fp, score_attr, {}).get(k, 0.5) for k in all_keys]
            for fp in fingerprints
        ]
    )

    fig, ax = plt.subplots(figsize=(max(10, len(all_keys) * 0.8), max(4, len(model_names) * 0.8)))
    sns.heatmap(
        data,
        xticklabels=[k.replace("_", " ").title() for k in all_keys],
        yticklabels=model_names,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig


def value_comparison_bar(
    fingerprints: Sequence[OntologicalFingerprint],
    *,
    title: str = "Value Hierarchy Comparison",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing axiological scores across models."""
    all_keys: list[str] = []
    seen: set[str] = set()
    for fp in fingerprints:
        for k in fp.axiology_scores:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if not all_keys:
        raise ValueError("No axiology scores to plot.")

    x = np.arange(len(all_keys))
    width = 0.8 / max(len(fingerprints), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(all_keys) * 1.2), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(fingerprints), 1)))

    for idx, fp in enumerate(fingerprints):
        values = [fp.axiology_scores.get(k, 0.5) for k in all_keys]
        offset = (idx - len(fingerprints) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=fp.model_name, color=colors[idx])

    ax.set_ylabel("Importance Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in all_keys], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig
