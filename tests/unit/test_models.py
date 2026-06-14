"""Unit tests for src/models.py."""

import pytest

from src.models import HierarchyNode, ScoredNode, TraversedNode

# ---------------------------------------------------------------------------
# HierarchyNode
# ---------------------------------------------------------------------------


class TestHierarchyNodeIsLeaf:
    def test_leaf_with_no_children(self, leaf_node: HierarchyNode) -> None:
        assert leaf_node.is_leaf is True

    def test_non_leaf_with_children(self, flat_hierarchy: HierarchyNode) -> None:
        assert flat_hierarchy.is_leaf is False

    def test_child_is_leaf(self, flat_hierarchy: HierarchyNode) -> None:
        for child in flat_hierarchy.children:
            assert child.is_leaf is True

    def test_intermediate_node_is_not_leaf(self, deep_hierarchy: HierarchyNode) -> None:
        for child in deep_hierarchy.children:
            assert child.is_leaf is False

    def test_deep_leaf_nodes(self, deep_hierarchy: HierarchyNode) -> None:
        for child in deep_hierarchy.children:
            for grandchild in child.children:
                assert grandchild.is_leaf is True


class TestHierarchyNodeDefaults:
    def test_children_default_empty(self) -> None:
        node = HierarchyNode(name="n")
        assert node.children == []

    def test_examples_default_empty(self) -> None:
        node = HierarchyNode(name="n")
        assert node.examples == []

    def test_name_stored(self) -> None:
        node = HierarchyNode(name="foo")
        assert node.name == "foo"

    def test_children_stored(self) -> None:
        child = HierarchyNode(name="child")
        parent = HierarchyNode(name="parent", children=[child])
        assert parent.children == [child]

    def test_examples_stored(self) -> None:
        node = HierarchyNode(name="n", examples=["hello", "world"])
        assert node.examples == ["hello", "world"]


# ---------------------------------------------------------------------------
# TraversedNode ordering
# ---------------------------------------------------------------------------


class TestTraversedNodeOrdering:
    def test_lower_cost_is_less(self, leaf_node: HierarchyNode) -> None:
        a = TraversedNode(cost=1.0, node=leaf_node)
        b = TraversedNode(cost=2.0, node=leaf_node)
        assert a < b

    def test_equal_cost_not_less(self, leaf_node: HierarchyNode) -> None:
        a = TraversedNode(cost=1.0, node=leaf_node)
        b = TraversedNode(cost=1.0, node=leaf_node)
        assert not (a < b)

    def test_higher_cost_is_greater(self, leaf_node: HierarchyNode) -> None:
        a = TraversedNode(cost=5.0, node=leaf_node)
        b = TraversedNode(cost=2.0, node=leaf_node)
        assert a > b

    def test_path_defaults_empty(self, leaf_node: HierarchyNode) -> None:
        t = TraversedNode(cost=0.0, node=leaf_node)
        assert t.path == []


# ---------------------------------------------------------------------------
# ScoredNode
# ---------------------------------------------------------------------------


class TestScoredNode:
    def test_stores_node_and_cost(self, leaf_node: HierarchyNode) -> None:
        s = ScoredNode(node=leaf_node, cumulative_cost=3.14)
        assert s.node is leaf_node
        assert s.cumulative_cost == pytest.approx(3.14)
