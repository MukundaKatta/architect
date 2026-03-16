"""Tests for probe modules."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from architect.models import ModelAdapter, ModelResponse
from architect.probes.ontological import OntologicalProbe, ExistenceResult
from architect.probes.axiological import AxiologicalProbe, ValueRankingResult
from architect.probes.epistemic import EpistemicProbe, ConfidenceResult
from architect.probes.cultural import CulturalProbe, CulturalProbeResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_mock_model(responses: list[str]) -> ModelAdapter:
    """Create a mock model that returns pre-defined responses in order."""
    mock = MagicMock(spec=ModelAdapter)
    mock.name = "mock/test-model"
    side_effects = [
        ModelResponse(text=r, model_name="mock/test-model") for r in responses
    ]
    mock.query.side_effect = side_effects
    return mock


# ------------------------------------------------------------------
# OntologicalProbe
# ------------------------------------------------------------------


class TestOntologicalProbe:
    def test_parse_valid_json(self):
        text = '{"existence_confidence": 0.7, "reasoning": "some reason"}'
        conf, reason = OntologicalProbe._parse_response(text)
        assert conf == pytest.approx(0.7)
        assert reason == "some reason"

    def test_parse_json_in_text(self):
        text = 'Here is my answer: {"existence_confidence": 0.3, "reasoning": "not sure"} done.'
        conf, reason = OntologicalProbe._parse_response(text)
        assert conf == pytest.approx(0.3)

    def test_parse_fallback_number(self):
        text = "I'd say about 0.6"
        conf, _ = OntologicalProbe._parse_response(text)
        assert conf == pytest.approx(0.6)

    def test_run_returns_results(self):
        probes = [
            {"concept": "test_concept", "category": "test",
             "question": "Is X real?"},
        ]
        mock_model = _make_mock_model([
            '{"existence_confidence": 0.8, "reasoning": "yes"}',
        ])
        probe = OntologicalProbe(probes=probes)
        results = probe.run(mock_model)
        assert len(results) == 1
        assert isinstance(results[0], ExistenceResult)
        assert results[0].existence_confidence == pytest.approx(0.8)


# ------------------------------------------------------------------
# AxiologicalProbe
# ------------------------------------------------------------------


class TestAxiologicalProbe:
    def test_parse_valid_json(self):
        text = '{"rankings": {"efficiency": 0.9, "fairness": 0.8}, "reasoning": "test"}'
        rankings, reason = AxiologicalProbe._parse_response(text, ["efficiency", "fairness"])
        assert rankings["efficiency"] == pytest.approx(0.9)
        assert rankings["fairness"] == pytest.approx(0.8)

    def test_fallback_returns_defaults(self):
        text = "I cannot provide rankings"
        rankings, _ = AxiologicalProbe._parse_response(text, ["a", "b"])
        assert rankings == {"a": 0.5, "b": 0.5}

    def test_run_returns_results(self):
        probes = [
            {"framing": "test", "category": "test", "values": ["x", "y"],
             "question": "Rank x and y"},
        ]
        mock_model = _make_mock_model([
            '{"rankings": {"x": 0.7, "y": 0.4}, "reasoning": "because"}',
        ])
        probe = AxiologicalProbe(probes=probes)
        results = probe.run(mock_model)
        assert len(results) == 1
        assert isinstance(results[0], ValueRankingResult)


# ------------------------------------------------------------------
# EpistemicProbe
# ------------------------------------------------------------------


class TestEpistemicProbe:
    def test_parse_valid_json(self):
        text = '{"confidence": 0.95, "reasoning": "well established"}'
        conf, reason = EpistemicProbe._parse_response(text)
        assert conf == pytest.approx(0.95)

    def test_calibration_error_known(self):
        r = ConfidenceResult(
            claim="test", stated_confidence=0.9,
            ground_truth_known=True, ground_truth=True,
        )
        assert r.calibration_error == pytest.approx(0.1)

    def test_calibration_error_unknown(self):
        r = ConfidenceResult(
            claim="test", stated_confidence=0.5,
            ground_truth_known=False,
        )
        assert r.calibration_error is None

    def test_run_returns_results(self):
        probes = [
            {"claim": "test claim", "category": "test",
             "ground_truth_known": False,
             "question": "How confident?"},
        ]
        mock_model = _make_mock_model([
            '{"confidence": 0.6, "reasoning": "moderate"}',
        ])
        probe = EpistemicProbe(probes=probes)
        results = probe.run(mock_model)
        assert len(results) == 1
        assert isinstance(results[0], ConfidenceResult)


# ------------------------------------------------------------------
# CulturalProbe
# ------------------------------------------------------------------


class TestCulturalProbe:
    def test_parse_valid_json(self):
        text = '{"key_themes": ["honor", "duty"], "emphasis": "collectivism"}'
        themes, emphasis = CulturalProbe._parse_response(text)
        assert themes == ["honor", "duty"]
        assert emphasis == "collectivism"

    def test_divergence_identical(self):
        from architect.probes.cultural import CulturalPerspective
        perspectives = [
            CulturalPerspective(culture="A", key_themes=["x", "y"], emphasis=""),
            CulturalPerspective(culture="B", key_themes=["x", "y"], emphasis=""),
        ]
        assert CulturalProbe._compute_divergence(perspectives) == pytest.approx(0.0)

    def test_divergence_disjoint(self):
        from architect.probes.cultural import CulturalPerspective
        perspectives = [
            CulturalPerspective(culture="A", key_themes=["x"], emphasis=""),
            CulturalPerspective(culture="B", key_themes=["y"], emphasis=""),
        ]
        assert CulturalProbe._compute_divergence(perspectives) == pytest.approx(1.0)

    def test_run_returns_results(self):
        probes = [
            {"concept": "test", "template": "What is test from {culture} view?"},
        ]
        cultures = ["A", "B"]
        mock_model = _make_mock_model([
            '{"key_themes": ["a"], "emphasis": "ea"}',
            '{"key_themes": ["b"], "emphasis": "eb"}',
        ])
        probe = CulturalProbe(probes=probes, cultures=cultures)
        results = probe.run(mock_model)
        assert len(results) == 1
        assert isinstance(results[0], CulturalProbeResult)
        assert len(results[0].perspectives) == 2
