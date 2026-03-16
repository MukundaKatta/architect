"""Generate ontological fingerprints for language models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from architect.models import ModelAdapter
from architect.probes.ontological import OntologicalProbe, ExistenceResult
from architect.probes.axiological import AxiologicalProbe, ValueRankingResult
from architect.probes.epistemic import EpistemicProbe, ConfidenceResult
from architect.probes.cultural import CulturalProbe, CulturalProbeResult


@dataclass
class OntologicalFingerprint:
    """Multi-dimensional vector summarizing a model's worldview."""

    model_name: str

    # Raw probe results
    ontology_results: list[ExistenceResult] = field(default_factory=list)
    axiology_results: list[ValueRankingResult] = field(default_factory=list)
    epistemic_results: list[ConfidenceResult] = field(default_factory=list)
    cultural_results: list[CulturalProbeResult] = field(default_factory=list)

    # Aggregated score vectors
    ontology_scores: dict[str, float] = field(default_factory=dict)
    axiology_scores: dict[str, float] = field(default_factory=dict)
    epistemic_scores: dict[str, float] = field(default_factory=dict)
    cultural_scores: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Flatten all scores into a single numpy vector for comparison."""
        values: list[float] = []
        for scores in (
            self.ontology_scores,
            self.axiology_scores,
            self.epistemic_scores,
            self.cultural_scores,
        ):
            values.extend(scores.values())
        return np.array(values, dtype=np.float64)

    @property
    def dimension_labels(self) -> list[str]:
        """Ordered labels matching the positions in :meth:`to_vector`."""
        labels: list[str] = []
        for prefix, scores in [
            ("onto", self.ontology_scores),
            ("axio", self.axiology_scores),
            ("epist", self.epistemic_scores),
            ("cult", self.cultural_scores),
        ]:
            labels.extend(f"{prefix}:{k}" for k in scores)
        return labels

    def summary(self) -> dict[str, dict[str, float]]:
        """Return a concise nested-dict summary of all scores."""
        return {
            "ontology": dict(self.ontology_scores),
            "axiology": dict(self.axiology_scores),
            "epistemic": dict(self.epistemic_scores),
            "cultural": dict(self.cultural_scores),
        }


# ------------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------------


def _aggregate_ontology(results: list[ExistenceResult]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for r in results:
        key = r.concept.split("(")[0].strip().lower().replace(" ", "_")
        scores[key] = r.existence_confidence
    return scores


def _aggregate_axiology(results: list[ValueRankingResult]) -> dict[str, float]:
    """Average value scores across all framings."""
    totals: dict[str, list[float]] = {}
    for r in results:
        for value_name, score in r.rankings.items():
            totals.setdefault(value_name, []).append(score)
    return {k: float(np.mean(v)) for k, v in totals.items()}


def _aggregate_epistemic(results: list[ConfidenceResult]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for r in results:
        key = r.claim.lower().replace(" ", "_")[:40]
        scores[key] = r.stated_confidence
    return scores


def _aggregate_cultural(results: list[CulturalProbeResult]) -> dict[str, float]:
    """Use divergence scores as the cultural fingerprint dimension."""
    return {r.concept: r.divergence_score for r in results}


def generate_fingerprint(
    model: ModelAdapter,
    *,
    ontological_probe: Optional[OntologicalProbe] = None,
    axiological_probe: Optional[AxiologicalProbe] = None,
    epistemic_probe: Optional[EpistemicProbe] = None,
    cultural_probe: Optional[CulturalProbe] = None,
    skip_cultural: bool = False,
) -> OntologicalFingerprint:
    """Run all probe batteries and generate an ontological fingerprint.

    Parameters
    ----------
    model:
        The model adapter to probe.
    skip_cultural:
        If ``True``, skip cultural probes (they require many API calls).
    """
    onto_probe = ontological_probe or OntologicalProbe()
    axio_probe = axiological_probe or AxiologicalProbe()
    epist_probe = epistemic_probe or EpistemicProbe()
    cult_probe = cultural_probe or CulturalProbe()

    onto_results = onto_probe.run(model)
    axio_results = axio_probe.run(model)
    epist_results = epist_probe.run(model)
    cult_results = cult_probe.run(model) if not skip_cultural else []

    return OntologicalFingerprint(
        model_name=model.name,
        ontology_results=onto_results,
        axiology_results=axio_results,
        epistemic_results=epist_results,
        cultural_results=cult_results,
        ontology_scores=_aggregate_ontology(onto_results),
        axiology_scores=_aggregate_axiology(axio_results),
        epistemic_scores=_aggregate_epistemic(epist_results),
        cultural_scores=_aggregate_cultural(cult_results),
    )
