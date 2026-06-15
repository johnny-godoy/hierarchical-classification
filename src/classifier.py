"""Define a classifier for hierarchical classification."""

import heapq
from typing import Self

import attrs

from src.models import HierarchyNode, NodeClassifier, TraversedNode
from src.scoring import NegLogScoringStrategy, ScoringStrategy


@attrs.define
class HierarchicalClassifier:
    """A classifier that traverses a hierarchy to classify an utterance.

    Attributes
    ----------
    scoring_strategy: ScoringStrategy
        The strategy used to score the children of each node during traversal.
    hierarchy: HierarchyNode
        The root of the hierarchy to classify against.
    """

    scoring_strategy: ScoringStrategy
    hierarchy: HierarchyNode

    @staticmethod
    def from_classifier(node_classifier: NodeClassifier, hierarchy: HierarchyNode) -> Self:
        """Construct a HierarchicalClassifier from a node classifier and hierarchy.

        Parameters
        ----------
        node_classifier: NodeClassifier
            A classifier that can predict probabilities for the children of each node.
        hierarchy: HierarchyNode
            The root of the hierarchy to classify against.

        Returns
        -------
        Self
            A HierarchicalClassifier instance that uses the given node classifier and hierarchy.
        """
        return HierarchicalClassifier(scoring_strategy=NegLogScoringStrategy(node_classifier), hierarchy=hierarchy)

    def classify(self, utterance: str) -> str:
        """Classify an utterance into the hierarchy.

        Parameters
        ----------
        utterance: str
            The text to classify.

        Returns
        -------
        str
            The name of the leaf node that best matches the utterance according to the scoring strategy.

        Raises
        ------
        ValueError
            If no leaf node is found in the hierarchy.
        """
        traversal_heap = [TraversedNode(cost=0.0, node=self.hierarchy)]
        while traversal_heap:
            current_node = heapq.heappop(traversal_heap)
            if current_node.node.is_leaf:
                return current_node.node.name
            for scored_node in self.scoring_strategy.score_children(
                utterance,
                current_node.node,
                current_node.cost,
            ):
                heapq.heappush(
                    traversal_heap,
                    TraversedNode(cost=scored_node.cumulative_cost, node=scored_node.node),
                )
        msg = "No leaf node found in the hierarchy."
        raise ValueError(msg)
