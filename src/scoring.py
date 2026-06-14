"""Scoring strategies for hierarchical classification."""

import itertools
from collections.abc import Iterator
from typing import Protocol, TypeVar

import numpy as np

from src.models import HierarchyNode, NodeClassifier, ScoredNode

T = TypeVar("T")


class ScoringStrategy(Protocol[T]):
    """Protocol for scoring children of a hierarchy node."""

    def score_children(
        self,
        utterance: str,
        node: HierarchyNode[T],
        current_cost: float,
    ) -> Iterator[ScoredNode]:
        """Score a node's children for the given utterance and cost."""


class NegLogScoringStrategy[T]:
    """Convert probabilities into cumulative negative-log costs."""

    def __init__(self, classifier: NodeClassifier[T]) -> None:
        self._classifier = classifier

    def score_children(
        self,
        utterance: str,
        node: HierarchyNode[T],
        current_cost: float,
    ) -> Iterator[ScoredNode]:
        probabilities = self._classifier.predict_proba(utterance, node)
        finite_mask = probabilities > 0
        if not np.any(finite_mask):
            return iter(())
        valid_scores = current_cost - np.log(probabilities[finite_mask])
        valid_children = itertools.compress(node.children, finite_mask)
        return (
            ScoredNode(node=child_node, cumulative_cost=score)
            for child_node, score in zip(valid_children, valid_scores)
        )
