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
        """Score children by converting probabilities to cumulative negative log costs.

        Parameters
        ----------
        utterance: str
            The text to classify.
        node: HierarchyNode
            The current hierarchy node whose children are to be scored.
        current_cost: float
            The cumulative cost up to the current node, used to compute cumulative
            costs for the children.

        Returns
        -------
            An iterator of ScoredNode instances for each child with finite probability.
        """
        probabilities = self._classifier.predict_proba(utterance, node)
        with np.errstate(divide="ignore", invalid="ignore"):
            costs = current_cost - np.log(probabilities)
        finite_mask = np.isfinite(costs)
        if not np.any(finite_mask):
            return iter(())
        valid_children = itertools.compress(node.children, finite_mask)
        return (
            ScoredNode(node=child_node, cumulative_cost=score)
            for child_node, score in zip(valid_children, costs[finite_mask], strict=False)
        )
