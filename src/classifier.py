"""Define a classifier for hierarchical classification."""

import dataclasses
import heapq
from typing import Self

from src.models import HierarchyNode, NodeClassifier, TraversedNode
from src.scoring import NegLogScoringStrategy, ScoringStrategy


@dataclasses.dataclass
class HierarchicalClassifier:
    scoring_strategy: ScoringStrategy
    hierarchy: HierarchyNode

    @staticmethod
    def from_classifier(node_classifier: NodeClassifier, hierarchy: HierarchyNode) -> Self:
        """Construct a HierarchicalClassifier from a node classifier and hierarchy."""
        return HierarchicalClassifier(scoring_strategy=NegLogScoringStrategy(node_classifier), hierarchy=hierarchy)

    def classify(self, utterance: str) -> str:
        """Classify an utterance into the hierarchy."""
        traversal_heap = [TraversedNode(cost=0.0, node=self.hierarchy, path=[])]
        best_leaf = None
        best_cost = float("inf")

        while traversal_heap:
            current_node = heapq.heappop(traversal_heap)
            if current_node.cost > best_cost:
                continue
            if current_node.node.is_leaf:
                if current_node.cost < best_cost:
                    best_cost = current_node.cost
                    best_leaf = current_node.node
                continue
            new_path = current_node.path + [current_node.node]
            for scored_node in self.scoring_strategy.score_children(
                utterance,
                current_node.node,
                current_node.cost,
            ):
                heapq.heappush(
                    traversal_heap,
                    TraversedNode(cost=scored_node.cumulative_cost, node=scored_node.node, path=new_path),
                )
        if best_leaf is None:
            raise ValueError("No leaf node found in the hierarchy.")
        return best_leaf.name
