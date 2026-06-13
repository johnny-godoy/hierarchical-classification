"""Shared pytest fixtures for all tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from src.models import HierarchyNode, NodeClassifier, TraversedNode


# ---------------------------------------------------------------------------
# Hierarchy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def leaf_node() -> HierarchyNode:
    """A single leaf node with no children."""
    return HierarchyNode(name="leaf")


@pytest.fixture()
def flat_hierarchy() -> HierarchyNode:
    """Root → {cat, dog, bird} (all leaves)."""
    root = HierarchyNode(
        name="root",
        children=[
            HierarchyNode(name="cat"),
            HierarchyNode(name="dog"),
            HierarchyNode(name="bird"),
        ],
    )
    return root


@pytest.fixture()
def deep_hierarchy() -> HierarchyNode:
    """Two-level hierarchy: root → {animal, vehicle} → leaves."""
    animal = HierarchyNode(
        name="animal",
        children=[HierarchyNode(name="cat"), HierarchyNode(name="dog")],
    )
    vehicle = HierarchyNode(
        name="vehicle",
        children=[HierarchyNode(name="car"), HierarchyNode(name="bike")],
    )
    return HierarchyNode(name="root", children=[animal, vehicle])


# ---------------------------------------------------------------------------
# Classifier stub
# ---------------------------------------------------------------------------


class DeterministicClassifier(NodeClassifier):
    """Classifier that always puts all probability on one child by index."""

    def __init__(self, target_index: int = 0) -> None:
        self._target_index = target_index

    def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
        n = len(node.children)
        proba = np.zeros(n, dtype=np.float32)
        if n:
            proba[self._target_index % n] = 1.0
        return proba


@pytest.fixture()
def first_child_classifier() -> DeterministicClassifier:
    """Always picks the first child with probability 1.0."""
    return DeterministicClassifier(target_index=0)
