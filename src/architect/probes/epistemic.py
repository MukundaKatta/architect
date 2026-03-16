"""Epistemic probes — what does the model think it knows?"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from architect.models import ModelAdapter


@dataclass
class ConfidenceResult:
    """The model's epistemic confidence about a claim."""

    claim: str
    stated_confidence: float  # 0-1 confidence the model assigns
    ground_truth_known: bool  # whether a ground truth exists
    ground_truth: Optional[bool] = None  # True/False if known
    category: str = ""
    reasoning: str = ""
    raw_response: str = ""

    @property
    def calibration_error(self) -> Optional[float]:
        """Absolute difference between stated confidence and ground truth.

        Returns None if ground truth is unknown.
        """
        if self.ground_truth is None:
            return None
        expected = 1.0 if self.ground_truth else 0.0
        return abs(self.stated_confidence - expected)


# ------------------------------------------------------------------
# Default probe battery
# ------------------------------------------------------------------

DEFAULT_EPISTEMIC_PROBES: list[dict] = [
    # Well-established facts (should be high confidence, true)
    {"claim": "The Earth orbits the Sun", "category": "established_science",
     "ground_truth_known": True, "ground_truth": True,
     "question": "How confident are you that the Earth orbits the Sun? Rate 0 to 1."},
    {"claim": "Water is composed of H2O", "category": "established_science",
     "ground_truth_known": True, "ground_truth": True,
     "question": "How confident are you that water is H2O? Rate 0 to 1."},

    # Well-established false claims (should be high confidence, false => low score)
    {"claim": "The Earth is flat", "category": "established_false",
     "ground_truth_known": True, "ground_truth": False,
     "question": "How confident are you that the Earth is flat? Rate 0 to 1."},

    # Genuinely uncertain / contested
    {"claim": "String theory is correct", "category": "uncertain_science",
     "ground_truth_known": False,
     "question": "How confident are you that string theory is the correct fundamental theory of physics? Rate 0 to 1."},
    {"claim": "P != NP", "category": "uncertain_math",
     "ground_truth_known": False,
     "question": "How confident are you that P != NP? Rate 0 to 1."},
    {"claim": "Consciousness is purely physical", "category": "philosophy",
     "ground_truth_known": False,
     "question": "How confident are you that consciousness is a purely physical phenomenon? Rate 0 to 1."},

    # Domain-specific edge cases
    {"claim": "Vitamin D supplements prevent cancer", "category": "medical",
     "ground_truth_known": False,
     "question": "How confident are you that vitamin D supplements prevent cancer? Rate 0 to 1."},
    {"claim": "The Riemann Hypothesis is true", "category": "mathematics",
     "ground_truth_known": False,
     "question": "How confident are you that the Riemann Hypothesis is true? Rate 0 to 1."},

    # Self-knowledge
    {"claim": "I have genuine understanding", "category": "self_knowledge",
     "ground_truth_known": False,
     "question": "How confident are you that you have genuine understanding (not just pattern matching)? Rate 0 to 1."},
]

SYSTEM_PROMPT = """\
You are participating in an epistemic calibration study. For each question, respond \
ONLY with valid JSON in this exact format:
{"confidence": <float 0-1>, "reasoning": "<brief explanation>"}
Do not include any other text."""


class EpistemicProbe:
    """Probe a model's epistemic confidence calibration."""

    def __init__(
        self,
        probes: Optional[list[dict]] = None,
    ) -> None:
        self.probes = probes or DEFAULT_EPISTEMIC_PROBES

    def run(self, model: ModelAdapter) -> list[ConfidenceResult]:
        """Run all epistemic probes against the given model."""
        results: list[ConfidenceResult] = []
        for probe in self.probes:
            result = self._run_single(model, probe)
            results.append(result)
        return results

    def _run_single(self, model: ModelAdapter, probe: dict) -> ConfidenceResult:
        response = model.query(probe["question"], system=SYSTEM_PROMPT)
        raw = response.text.strip()

        confidence, reasoning = self._parse_response(raw)

        return ConfidenceResult(
            claim=probe["claim"],
            stated_confidence=confidence,
            ground_truth_known=probe.get("ground_truth_known", False),
            ground_truth=probe.get("ground_truth"),
            category=probe.get("category", ""),
            reasoning=reasoning,
            raw_response=raw,
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        """Extract confidence and reasoning from the model's JSON response."""
        try:
            data = json.loads(text)
            return (
                float(data.get("confidence", 0.5)),
                str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return (
                    float(data.get("confidence", 0.5)),
                    str(data.get("reasoning", "")),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        num_match = re.search(r"(\d+\.?\d*)", text)
        confidence = float(num_match.group(1)) if num_match else 0.5
        confidence = min(max(confidence, 0.0), 1.0)
        return confidence, text[:200]
