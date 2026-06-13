"""Unit tests for src/models.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import HierarchyNode, NodeClassifier, ScoredNode, TraversedNode
from tests.conftest import DeterministicClassifier


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


class TestHierarchyNodeSaveLoad:
    @pytest.mark.anyio
    async def test_save_then_load_roundtrip(self, tmp_path, flat_hierarchy: HierarchyNode) -> None:
        path = tmp_path / "hierarchy.json"
        await flat_hierarchy.save(path)
        assert path.exists()
        loaded = await HierarchyNode.load(path)
        assert loaded.name == flat_hierarchy.name
        assert len(loaded.children) == len(flat_hierarchy.children)
        assert [c.name for c in loaded.children] == [c.name for c in flat_hierarchy.children]

    @pytest.mark.anyio
    async def test_save_creates_parent_dirs(self, tmp_path, leaf_node: HierarchyNode) -> None:
        path = tmp_path / "nested" / "deep" / "hierarchy.json"
        await leaf_node.save(path)
        assert path.exists()

    @pytest.mark.anyio
    async def test_save_produces_valid_json(self, tmp_path, leaf_node: HierarchyNode) -> None:
        import orjson

        path = tmp_path / "node.json"
        await leaf_node.save(path)
        data = orjson.loads(path.read_bytes())
        assert data["name"] == "leaf"
        assert data["children"] == []

    @pytest.mark.anyio
    async def test_load_preserves_examples(self, tmp_path) -> None:
        node = HierarchyNode(name="n", examples=["a", "b"])
        path = tmp_path / "node.json"
        await node.save(path)
        loaded = await HierarchyNode.load(path)
        assert loaded.examples == ["a", "b"]

    @pytest.mark.anyio
    async def test_load_reconstructs_children_as_hierarchy_nodes(self, tmp_path, flat_hierarchy) -> None:
        path = tmp_path / "flat.json"
        await flat_hierarchy.save(path)
        loaded = await HierarchyNode.load(path)
        for child in loaded.children:
            assert isinstance(child, HierarchyNode)


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


# ---------------------------------------------------------------------------
# NodeClassifier.neg_log_proba (default protocol implementation)
# ---------------------------------------------------------------------------


class TestNegLogProba:
    def _make_traversed(self, node: HierarchyNode, cost: float = 0.0) -> TraversedNode:
        return TraversedNode(cost=cost, node=node, path=[])

    def test_returns_scored_nodes_for_all_positive_children(self, flat_hierarchy: HierarchyNode) -> None:
        clf = DeterministicClassifier(target_index=0)
        traversed = self._make_traversed(flat_hierarchy)
        results = list(clf.neg_log_proba("test", traversed))
        # Only the first child gets probability 1.0, others get 0.0 → only one result
        assert len(results) == 1
        assert results[0].node.name == "cat"

    def test_neg_log_of_one_is_zero(self, flat_hierarchy: HierarchyNode) -> None:
        clf = DeterministicClassifier(target_index=0)
        traversed = self._make_traversed(flat_hierarchy)
        results = list(clf.neg_log_proba("test", traversed))
        assert results[0].cumulative_cost == pytest.approx(0.0)

    def test_cost_accumulates(self, flat_hierarchy: HierarchyNode) -> None:
        clf = DeterministicClassifier(target_index=0)
        traversed = self._make_traversed(flat_hierarchy, cost=5.0)
        results = list(clf.neg_log_proba("test", traversed))
        # -log(1.0) = 0, so cumulative = 5.0 + 0.0 = 5.0
        assert results[0].cumulative_cost == pytest.approx(5.0)

    def test_all_zero_proba_returns_empty(self, flat_hierarchy: HierarchyNode) -> None:
        class ZeroClassifier(NodeClassifier):
            def predict_proba(self, utterance, node):
                return np.zeros(len(node.children), dtype=np.float32)

        clf = ZeroClassifier()
        traversed = self._make_traversed(flat_hierarchy)
        results = list(clf.neg_log_proba("test", traversed))
        assert results == []

    def test_partial_positive_proba(self, flat_hierarchy: HierarchyNode) -> None:
        class PartialClassifier(NodeClassifier):
            def predict_proba(self, utterance, node):
                # 0.5, 0.5, 0.0
                proba = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                return proba

        clf = PartialClassifier()
        traversed = self._make_traversed(flat_hierarchy)
        results = list(clf.neg_log_proba("test", traversed))
        assert len(results) == 2
        names = {r.node.name for r in results}
        assert names == {"cat", "dog"}
        for r in results:
            assert r.cumulative_cost == pytest.approx(-np.log(0.5))
