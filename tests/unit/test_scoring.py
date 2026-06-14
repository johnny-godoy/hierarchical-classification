"""Unit tests for src/scoring.py."""

import numpy as np
import pytest

from src.models import HierarchyNode, NodeClassifier
from src.scoring import NegLogScoringStrategy
from tests.conftest import DeterministicClassifier


class TestNegLogScoringStrategy:
    def test_returns_scored_nodes_for_all_positive_children(self, flat_hierarchy: HierarchyNode) -> None:
        clf = NegLogScoringStrategy(DeterministicClassifier(target_index=0))
        results = list(clf.score_children("test", flat_hierarchy, current_cost=0.0))
        assert len(results) == 1
        assert results[0].node.name == "cat"

    def test_neg_log_of_one_is_zero(self, flat_hierarchy: HierarchyNode) -> None:
        clf = NegLogScoringStrategy(DeterministicClassifier(target_index=0))
        results = list(clf.score_children("test", flat_hierarchy, current_cost=0.0))
        assert results[0].cumulative_cost == pytest.approx(0.0)

    def test_cost_accumulates(self, flat_hierarchy: HierarchyNode) -> None:
        clf = NegLogScoringStrategy(DeterministicClassifier(target_index=0))
        results = list(clf.score_children("test", flat_hierarchy, current_cost=5.0))
        assert results[0].cumulative_cost == pytest.approx(5.0)

    def test_all_zero_proba_returns_empty(self, flat_hierarchy: HierarchyNode) -> None:
        class ZeroClassifier(NodeClassifier):
            def predict_proba(self, _: str, node: HierarchyNode) -> np.ndarray:
                return np.zeros(len(node.children), dtype=np.float32)

        clf = NegLogScoringStrategy(ZeroClassifier())
        results = list(clf.score_children("test", flat_hierarchy, current_cost=0.0))
        assert results == []

    def test_partial_positive_proba(self, flat_hierarchy: HierarchyNode) -> None:
        class PartialClassifier(NodeClassifier):
            def predict_proba(self, _: str, __: HierarchyNode) -> np.ndarray:
                return np.array([0.5, 0.5, 0.0], dtype=np.float32)

        clf = NegLogScoringStrategy(PartialClassifier())
        results = list(clf.score_children("test", flat_hierarchy, current_cost=0.0))
        assert len(results) == 2
        names = {r.node.name for r in results}
        assert names == {"cat", "dog"}
        for result in results:
            assert result.cumulative_cost == pytest.approx(-np.log(0.5))
