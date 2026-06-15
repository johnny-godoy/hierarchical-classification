"""Define models for hierarchical classification."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, TypeVar

import attrs

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

T = TypeVar("T")


class HierarchyNodeDict(TypedDict):
    """Typed representation for serialized hierarchy nodes."""

    name: str
    children: NotRequired[list[HierarchyNodeDict]]
    examples: NotRequired[list[object]]


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
    def from_dict(data: Mapping[str, object], *, _path: str = "root") -> HierarchyNode:
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

        Raises
        ------
        KeyError
            If a required key is missing.
        TypeError
            If any node field has an invalid type.
        """
        if "name" not in data:
            msg = f"{_path} is missing required key 'name'."
            raise KeyError(msg)

        name = data["name"]
        if not isinstance(name, str):
            msg = f"{_path}.name must be a string, got {type(name).__name__}."
            raise TypeError(msg)

        children_data = data.get("children", [])
        if not isinstance(children_data, Sequence) or isinstance(children_data, str):
            msg = f"{_path}.children must be a list, got {type(children_data).__name__}."
            raise TypeError(msg)

        examples_data = data.get("examples", [])
        if not isinstance(examples_data, Sequence) or isinstance(examples_data, str):
            msg = f"{_path}.examples must be a list, got {type(examples_data).__name__}."
            raise TypeError(msg)

        children: list[HierarchyNode] = []
        for index, child in enumerate(children_data):
            if not isinstance(child, Mapping):
                msg = f"{_path}.children[{index}] must be a mapping, got {type(child).__name__}."
                raise TypeError(msg)
            children.append(HierarchyNode.from_dict(child, _path=f"{_path}.children[{index}]"))

        return HierarchyNode(
            name=name,
            children=children,
            examples=examples_data,
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
