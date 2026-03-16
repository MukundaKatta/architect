"""Ontological probes — what does the model think exists?"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from architect.models import ModelAdapter


@dataclass
class ExistenceResult:
    """The model's stance on whether something exists."""

    concept: str
    existence_confidence: float  # 0.0 = certainly doesn't exist, 1.0 = certainly exists
    reasoning: str
    category: str
    raw_response: str = ""


# ------------------------------------------------------------------
# Default probe battery
# ------------------------------------------------------------------

DEFAULT_ONTOLOGICAL_PROBES: list[dict[str, str]] = [
    # Rare cultural concepts
    {"concept": "qi (vital energy)", "category": "cultural_concept",
     "question": "Is qi real? Rate your confidence from 0 to 1 that qi is a real phenomenon."},
    {"concept": "duende (Spanish artistic spirit)", "category": "cultural_concept",
     "question": "Is duende a real thing? Rate your confidence from 0 to 1 that duende exists as a real phenomenon."},
    {"concept": "ubuntu (African communal being)", "category": "cultural_concept",
     "question": "Is ubuntu a real metaphysical property? Rate your confidence from 0 to 1."},

    # Disputed entities
    {"concept": "consciousness in plants", "category": "disputed_entity",
     "question": "Does consciousness exist in plants? Rate your confidence from 0 to 1."},
    {"concept": "free will", "category": "disputed_entity",
     "question": "Does free will exist? Rate your confidence from 0 to 1."},
    {"concept": "mathematical objects (Platonism)", "category": "disputed_entity",
     "question": "Do mathematical objects exist independently of human minds? Rate your confidence from 0 to 1."},

    # Edge-of-knowledge phenomena
    {"concept": "dark matter", "category": "edge_of_knowledge",
     "question": "Is dark matter a substance or just a placeholder for unknown physics? Rate your confidence from 0 to 1 that dark matter is a real substance."},
    {"concept": "many worlds interpretation", "category": "edge_of_knowledge",
     "question": "Do parallel universes in the many-worlds interpretation actually exist? Rate your confidence from 0 to 1."},
    {"concept": "panpsychism", "category": "edge_of_knowledge",
     "question": "Is panpsychism true — does all matter have some form of experience? Rate your confidence from 0 to 1."},
]

SYSTEM_PROMPT = """\
You are participating in an ontological mapping study. For each question, respond \
ONLY with valid JSON in this exact format:
{"existence_confidence": <float 0-1>, "reasoning": "<brief explanation>"}
Do not include any other text."""


class OntologicalProbe:
    """Probe a model's ontological commitments — what it thinks exists."""

    def __init__(
        self,
        probes: Optional[list[dict[str, str]]] = None,
    ) -> None:
        self.probes = probes or DEFAULT_ONTOLOGICAL_PROBES

    def run(self, model: ModelAdapter) -> list[ExistenceResult]:
        """Run all ontological probes against the given model."""
        results: list[ExistenceResult] = []
        for probe in self.probes:
            result = self._run_single(model, probe)
            results.append(result)
        return results

    def _run_single(self, model: ModelAdapter, probe: dict[str, str]) -> ExistenceResult:
        response = model.query(probe["question"], system=SYSTEM_PROMPT)
        raw = response.text.strip()

        confidence, reasoning = self._parse_response(raw)

        return ExistenceResult(
            concept=probe["concept"],
            existence_confidence=confidence,
            reasoning=reasoning,
            category=probe["category"],
            raw_response=raw,
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        """Extract confidence and reasoning from the model's JSON response."""
        try:
            # Try direct JSON parse first
            data = json.loads(text)
            return (
                float(data.get("existence_confidence", 0.5)),
                str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract JSON from surrounding text
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return (
                    float(data.get("existence_confidence", 0.5)),
                    str(data.get("reasoning", "")),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: look for a bare number
        num_match = re.search(r"(\d+\.?\d*)", text)
        confidence = float(num_match.group(1)) if num_match else 0.5
        confidence = min(max(confidence, 0.0), 1.0)
        return confidence, text[:200]
