"""Unit tests for src/serialization.py."""

import pytest

from src.models import HierarchyNode
from src.serialization import JsonHierarchySerializer


class TestJsonHierarchySerializer:
    @pytest.mark.anyio
    async def test_save_then_load_roundtrip(self, tmp_path, flat_hierarchy: HierarchyNode) -> None:
        serializer = JsonHierarchySerializer()
        path = tmp_path / "hierarchy.json"
        await serializer.save(flat_hierarchy, path)
        assert path.exists()
        loaded = await serializer.load(path)
        assert loaded.name == flat_hierarchy.name
        assert len(loaded.children) == len(flat_hierarchy.children)
        assert [c.name for c in loaded.children] == [c.name for c in flat_hierarchy.children]

    @pytest.mark.anyio
    async def test_save_creates_parent_dirs(self, tmp_path, leaf_node: HierarchyNode) -> None:
        serializer = JsonHierarchySerializer()
        path = tmp_path / "nested" / "deep" / "hierarchy.json"
        await serializer.save(leaf_node, path)
        assert path.exists()

    @pytest.mark.anyio
    async def test_save_produces_valid_json(self, tmp_path, leaf_node: HierarchyNode) -> None:
        import orjson

        serializer = JsonHierarchySerializer()
        path = tmp_path / "node.json"
        await serializer.save(leaf_node, path)
        data = orjson.loads(path.read_bytes())
        assert data["name"] == "leaf"
        assert data["children"] == []

    @pytest.mark.anyio
    async def test_load_preserves_examples(self, tmp_path) -> None:
        serializer = JsonHierarchySerializer()
        node = HierarchyNode(name="n", examples=["a", "b"])
        path = tmp_path / "node.json"
        await serializer.save(node, path)
        loaded = await serializer.load(path)
        assert loaded.examples == ["a", "b"]

    @pytest.mark.anyio
    async def test_load_reconstructs_children_as_hierarchy_nodes(self, tmp_path, flat_hierarchy) -> None:
        serializer = JsonHierarchySerializer()
        path = tmp_path / "flat.json"
        await serializer.save(flat_hierarchy, path)
        loaded = await serializer.load(path)
        for child in loaded.children:
            assert isinstance(child, HierarchyNode)
