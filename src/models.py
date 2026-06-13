"""Define models for hierarchical classification."""

from __future__ import annotations

import functools
import itertools
import os
from collections.abc import Iterator
from typing import Protocol, TypeVar

import anyio
import attrs
import numpy as np
import numpy.typing as npt
import orjson

T = TypeVar("T")
PathLike = str | os.PathLike


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

    async def save(self, path: PathLike) -> None:
        """Save the hierarchy to a file."""
        path = anyio.Path(path)
        await path.parent.mkdir(parents=True, exist_ok=True)
        await path.write_bytes(orjson.dumps(attrs.asdict(self), option=orjson.OPT_INDENT_2))

    async def load(path: PathLike) -> HierarchyNode[T]:
        """Load the hierarchy from a file."""
        data = orjson.loads(await anyio.Path(path).read_bytes())
        return HierarchyNode._from_dict(data)

    @staticmethod
    def _from_dict(data: dict) -> HierarchyNode:
        """Recursively reconstruct a HierarchyNode from a plain dict."""
        return HierarchyNode(
            name=data["name"],
            children=[HierarchyNode._from_dict(c) for c in data.get("children", [])],
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

    def neg_log_proba(
        self,
        utterance: str,
        node: TraversedNode,
    ) -> Iterator[ScoredNode]:
        """Predict the negative log probabilities of the utterance belonging to each class."""
        probabilities = self.predict_proba(utterance, node.node)
        finite_mask = probabilities > 0
        if not np.any(finite_mask):
            return iter(())
        valid_scores = node.cost - np.log(probabilities[finite_mask])
        valid_children = itertools.compress(node.node.children, finite_mask)
        return (
            ScoredNode(node=child_node, cumulative_cost=score)
            for child_node, score in zip(valid_children, valid_scores)
        )
