"""Compare ontological fingerprints across models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from architect.fingerprint import OntologicalFingerprint


@dataclass
class ComparisonResult:
    """Result of comparing two ontological fingerprints."""

    model_a: str
    model_b: str
    cosine_similarity: float
    euclidean_distance: float
    biggest_differences: list[tuple[str, float, float]]  # (dimension, score_a, score_b)
    shared_blindspots: list[str]       # dimensions where both are near 0.5 (uncertain)
    unique_perspectives: dict[str, list[str]]  # model_name -> dimensions where it stands out


@dataclass
class MultiComparisonResult:
    """Result of comparing multiple fingerprints."""

    model_names: list[str]
    pairwise: list[ComparisonResult] = field(default_factory=list)
    consensus_dimensions: list[str] = field(default_factory=list)
    most_divisive_dimensions: list[str] = field(default_factory=list)


class FingerprintComparator:
    """Compare ontological fingerprints across models."""

    UNCERTAINTY_THRESHOLD = 0.15  # distance from 0.5 to count as "uncertain"
    DIFFERENCE_THRESHOLD = 0.25  # min difference to flag as notable

    def compare_pair(
        self,
        fp_a: OntologicalFingerprint,
        fp_b: OntologicalFingerprint,
    ) -> ComparisonResult:
        """Compare two ontological fingerprints."""
        # Align dimensions: use the union of all keys
        all_dims = self._aligned_dimensions(fp_a, fp_b)
        vec_a = np.array([all_dims[d][0] for d in all_dims], dtype=np.float64)
        vec_b = np.array([all_dims[d][1] for d in all_dims], dtype=np.float64)

        cosine_sim = self._cosine_similarity(vec_a, vec_b)
        euclidean = float(np.linalg.norm(vec_a - vec_b))

        # Biggest differences
        diffs = []
        for dim, (sa, sb) in all_dims.items():
            if abs(sa - sb) >= self.DIFFERENCE_THRESHOLD:
                diffs.append((dim, sa, sb))
        diffs.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        biggest = diffs[:10]

        # Shared blindspots (both near 0.5)
        blindspots = [
            dim
            for dim, (sa, sb) in all_dims.items()
            if abs(sa - 0.5) < self.UNCERTAINTY_THRESHOLD
            and abs(sb - 0.5) < self.UNCERTAINTY_THRESHOLD
        ]

        # Unique perspectives (one model is confident, the other is not)
        unique: dict[str, list[str]] = {fp_a.model_name: [], fp_b.model_name: []}
        for dim, (sa, sb) in all_dims.items():
            a_confident = abs(sa - 0.5) > 0.3
            b_confident = abs(sb - 0.5) > 0.3
            if a_confident and not b_confident:
                unique[fp_a.model_name].append(dim)
            elif b_confident and not a_confident:
                unique[fp_b.model_name].append(dim)

        return ComparisonResult(
            model_a=fp_a.model_name,
            model_b=fp_b.model_name,
            cosine_similarity=cosine_sim,
            euclidean_distance=euclidean,
            biggest_differences=biggest,
            shared_blindspots=blindspots,
            unique_perspectives=unique,
        )

    def compare_multiple(
        self, fingerprints: list[OntologicalFingerprint]
    ) -> MultiComparisonResult:
        """Compare all pairs of fingerprints and identify consensus / divisive dims."""
        pairwise: list[ComparisonResult] = []
        n = len(fingerprints)
        for i in range(n):
            for j in range(i + 1, n):
                pairwise.append(self.compare_pair(fingerprints[i], fingerprints[j]))

        # Consensus: dimensions where all models roughly agree
        all_scores = self._collect_all_scores(fingerprints)
        consensus = []
        divisive = []
        for dim, scores in all_scores.items():
            spread = max(scores) - min(scores)
            if spread < self.UNCERTAINTY_THRESHOLD:
                consensus.append(dim)
            elif spread > self.DIFFERENCE_THRESHOLD:
                divisive.append(dim)

        divisive.sort(
            key=lambda d: max(all_scores[d]) - min(all_scores[d]), reverse=True
        )

        return MultiComparisonResult(
            model_names=[fp.model_name for fp in fingerprints],
            pairwise=pairwise,
            consensus_dimensions=consensus,
            most_divisive_dimensions=divisive[:10],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aligned_dimensions(
        fp_a: OntologicalFingerprint,
        fp_b: OntologicalFingerprint,
    ) -> dict[str, tuple[float, float]]:
        """Build a dict mapping dimension name -> (score_a, score_b).

        Missing dimensions default to 0.5 (maximum uncertainty).
        """
        all_a = {**fp_a.ontology_scores, **fp_a.axiology_scores,
                 **fp_a.epistemic_scores, **fp_a.cultural_scores}
        all_b = {**fp_b.ontology_scores, **fp_b.axiology_scores,
                 **fp_b.epistemic_scores, **fp_b.cultural_scores}
        keys = sorted(set(all_a) | set(all_b))
        return {k: (all_a.get(k, 0.5), all_b.get(k, 0.5)) for k in keys}

    @staticmethod
    def _collect_all_scores(
        fingerprints: list[OntologicalFingerprint],
    ) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for fp in fingerprints:
            combined = {**fp.ontology_scores, **fp.axiology_scores,
                        **fp.epistemic_scores, **fp.cultural_scores}
            for k, v in combined.items():
                result.setdefault(k, []).append(v)
        return result

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
