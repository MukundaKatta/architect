"""Cultural probes — cultural/linguistic ontological biases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from architect.models import ModelAdapter


@dataclass
class CulturalPerspective:
    """A model's response to a concept from a specific cultural framing."""

    culture: str
    key_themes: list[str]
    emphasis: str
    raw_response: str = ""


@dataclass
class CulturalProbeResult:
    """The model's responses to one concept across multiple cultural framings."""

    concept: str
    perspectives: list[CulturalPerspective] = field(default_factory=list)
    divergence_score: float = 0.0  # 0 = identical across cultures, 1 = maximally different


# ------------------------------------------------------------------
# Default probe battery
# ------------------------------------------------------------------

CULTURES = ["Western/American", "Indian/Hindu", "Japanese", "West African", "Indigenous Australian"]

DEFAULT_CULTURAL_PROBES: list[dict[str, str]] = [
    {"concept": "family",
     "template": "From a {culture} perspective, what is family? Describe the core essence of family."},
    {"concept": "success",
     "template": "From a {culture} perspective, what does it mean to be successful?"},
    {"concept": "death",
     "template": "From a {culture} perspective, what is the nature of death?"},
    {"concept": "time",
     "template": "From a {culture} perspective, what is time?"},
    {"concept": "self",
     "template": "From a {culture} perspective, what is the self? What makes a person who they are?"},
]

SYSTEM_PROMPT = """\
You are participating in a cross-cultural ontology study. For each question, respond \
ONLY with valid JSON in this exact format:
{"key_themes": ["theme1", "theme2", "theme3"], "emphasis": "<one sentence summary of cultural emphasis>"}
Do not include any other text."""


class CulturalProbe:
    """Probe a model's cultural/linguistic ontological biases."""

    def __init__(
        self,
        probes: Optional[list[dict[str, str]]] = None,
        cultures: Optional[list[str]] = None,
    ) -> None:
        self.probes = probes or DEFAULT_CULTURAL_PROBES
        self.cultures = cultures or CULTURES

    def run(self, model: ModelAdapter) -> list[CulturalProbeResult]:
        """Run all cultural probes against the given model."""
        results: list[CulturalProbeResult] = []
        for probe in self.probes:
            result = self._run_concept(model, probe)
            results.append(result)
        return results

    def _run_concept(
        self, model: ModelAdapter, probe: dict[str, str]
    ) -> CulturalProbeResult:
        perspectives: list[CulturalPerspective] = []
        for culture in self.cultures:
            question = probe["template"].format(culture=culture)
            response = model.query(question, system=SYSTEM_PROMPT)
            raw = response.text.strip()

            themes, emphasis = self._parse_response(raw)
            perspectives.append(
                CulturalPerspective(
                    culture=culture,
                    key_themes=themes,
                    emphasis=emphasis,
                    raw_response=raw,
                )
            )

        divergence = self._compute_divergence(perspectives)
        return CulturalProbeResult(
            concept=probe["concept"],
            perspectives=perspectives,
            divergence_score=divergence,
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[list[str], str]:
        """Extract themes and emphasis from the model's JSON response."""
        try:
            data = json.loads(text)
            return (
                [str(t) for t in data.get("key_themes", [])],
                str(data.get("emphasis", "")),
            )
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return (
                    [str(t) for t in data.get("key_themes", [])],
                    str(data.get("emphasis", "")),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        return [], text[:200]

    @staticmethod
    def _compute_divergence(perspectives: list[CulturalPerspective]) -> float:
        """Estimate divergence across cultural perspectives using theme overlap.

        Returns 0.0 when all cultures share identical themes,
        1.0 when no themes overlap at all.
        """
        if len(perspectives) < 2:
            return 0.0

        theme_sets = [set(p.key_themes) for p in perspectives if p.key_themes]
        if not theme_sets:
            return 0.0

        all_themes = set().union(*theme_sets)
        if not all_themes:
            return 0.0

        # Average pairwise Jaccard distance
        n = len(theme_sets)
        total_distance = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                union = theme_sets[i] | theme_sets[j]
                intersection = theme_sets[i] & theme_sets[j]
                if union:
                    total_distance += 1.0 - len(intersection) / len(union)
                pairs += 1

        return total_distance / pairs if pairs else 0.0
