"""Define a classifier for hierarchical classification."""

import dataclasses
import heapq

from src.models import HierarchyNode, NodeClassifier, TraversedNode


@dataclasses.dataclass
class HierarchicalClassifier:
    classifier: NodeClassifier
    hierarchy: HierarchyNode

    def classify(self, utterance: str) -> str:
        """Classify an utterance into the hierarchy."""
        traversal_heap = [TraversedNode(cost=0.0, node=self.hierarchy, path=[])]
        best_leaf = None
        best_cost = float("inf")

        while traversal_heap:
            current_node = heapq.heappop(traversal_heap)
            if current_node.cost > best_cost:
                continue
            new_path = current_node.path + [current_node.node]
            if current_node.node.is_leaf:
                if current_node.cost < best_cost:
                    best_cost = current_node.cost
                    best_leaf = current_node.node
                    continue
            for scored_node in self.classifier.neg_log_proba(
                utterance,
                current_node,
            ):
                heapq.heappush(
                    traversal_heap,
                    TraversedNode(cost=scored_node.cumulative_cost, node=scored_node.node, path=new_path),
                )
        if best_leaf is None:
            raise ValueError("No leaf node found in the hierarchy.")
        return best_leaf.name
