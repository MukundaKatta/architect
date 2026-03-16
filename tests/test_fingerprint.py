"""Tests for fingerprint generation and comparison."""

from __future__ import annotations

import numpy as np
import pytest

from architect.fingerprint import OntologicalFingerprint
from architect.comparator import FingerprintComparator


# ------------------------------------------------------------------
# OntologicalFingerprint
# ------------------------------------------------------------------


class TestOntologicalFingerprint:
    def _make_fp(self, name: str, onto: dict, axio: dict, epist: dict, cult: dict):
        return OntologicalFingerprint(
            model_name=name,
            ontology_scores=onto,
            axiology_scores=axio,
            epistemic_scores=epist,
            cultural_scores=cult,
        )

    def test_to_vector(self):
        fp = self._make_fp(
            "test",
            {"a": 0.1, "b": 0.2},
            {"x": 0.3},
            {"p": 0.4},
            {"c": 0.5},
        )
        vec = fp.to_vector()
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 5
        np.testing.assert_array_almost_equal(vec, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_dimension_labels(self):
        fp = self._make_fp(
            "test",
            {"a": 0.1},
            {"x": 0.2},
            {"p": 0.3},
            {"c": 0.4},
        )
        labels = fp.dimension_labels
        assert labels == ["onto:a", "axio:x", "epist:p", "cult:c"]

    def test_summary(self):
        fp = self._make_fp("test", {"a": 0.1}, {"x": 0.2}, {}, {})
        s = fp.summary()
        assert s["ontology"] == {"a": 0.1}
        assert s["axiology"] == {"x": 0.2}


# ------------------------------------------------------------------
# FingerprintComparator
# ------------------------------------------------------------------


class TestFingerprintComparator:
    def _make_fp(self, name: str, scores: dict[str, float]):
        return OntologicalFingerprint(
            model_name=name,
            ontology_scores=scores,
        )

    def test_identical_fingerprints(self):
        fp_a = self._make_fp("a", {"x": 0.5, "y": 0.8})
        fp_b = self._make_fp("b", {"x": 0.5, "y": 0.8})
        comp = FingerprintComparator()
        result = comp.compare_pair(fp_a, fp_b)
        assert result.cosine_similarity == pytest.approx(1.0)
        assert result.euclidean_distance == pytest.approx(0.0)
        assert len(result.biggest_differences) == 0

    def test_different_fingerprints(self):
        fp_a = self._make_fp("a", {"x": 0.0, "y": 1.0})
        fp_b = self._make_fp("b", {"x": 1.0, "y": 0.0})
        comp = FingerprintComparator()
        result = comp.compare_pair(fp_a, fp_b)
        assert result.cosine_similarity < 0.1
        assert result.euclidean_distance > 1.0
        assert len(result.biggest_differences) == 2

    def test_shared_blindspots(self):
        fp_a = self._make_fp("a", {"x": 0.5, "y": 0.5})
        fp_b = self._make_fp("b", {"x": 0.5, "y": 0.5})
        comp = FingerprintComparator()
        result = comp.compare_pair(fp_a, fp_b)
        assert "x" in result.shared_blindspots
        assert "y" in result.shared_blindspots

    def test_compare_multiple(self):
        fps = [
            self._make_fp("a", {"x": 0.1, "y": 0.9}),
            self._make_fp("b", {"x": 0.2, "y": 0.8}),
            self._make_fp("c", {"x": 0.9, "y": 0.1}),
        ]
        comp = FingerprintComparator()
        result = comp.compare_multiple(fps)
        assert len(result.pairwise) == 3  # C(3,2)
        assert len(result.model_names) == 3
