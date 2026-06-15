"""Define models for hierarchical classification."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol, TypeVar

import attrs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import numpy.typing as npt

T = TypeVar("T")


@attrs.define
class HierarchyNode[T]:
    """A node in the hierarchy."""

    name: str
    children: list[HierarchyNode[T]] = attrs.field(factory=list)
    examples: list[T] = attrs.field(factory=list)

    @functools.cached_property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return not self.children

    @staticmethod
    def from_dict(data: Mapping) -> HierarchyNode:
        """Recursively reconstruct a HierarchyNode from a plain dict.

        Parameters
        ----------
        data: dict
            A dictionary with keys "name", "children", and "examples" representing
            a hierarchy node. The "children" key should be a list of similar
            dictionaries for child nodes.

        Returns
        -------
            A HierarchyNode instance reconstructed from the input dictionary.
        """
        return HierarchyNode(
            name=data["name"],
            children=[HierarchyNode.from_dict(c) for c in data.get("children", [])],
            examples=data.get("examples", []),
        )


@attrs.define(order=True)
class TraversedNode[T]:
    """A node that has been traversed during classification."""

    cost: float
    node: HierarchyNode[T] = attrs.field(order=False)
    path: list[HierarchyNode[T]] = attrs.field(factory=list, order=False)


@attrs.define
class ScoredNode:
    """A dataclass to hold all classes that have finite logproba scores."""

    node: HierarchyNode
    cumulative_cost: float


class NodeClassifier(Protocol[T]):
    """Protocol for a node classifier."""

    def predict_proba(
        self,
        utterance: str,
        node: HierarchyNode,
    ) -> npt.NDArray[np.float32]:
        """Predict the probabilities of the utterance belonging to each class."""
