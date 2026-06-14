"""Serialization helpers for hierarchy nodes."""

import os
from typing import Protocol, TypeVar

import anyio
import attrs
import orjson

from src.models import HierarchyNode

T = TypeVar("T")
PathLike = str | os.PathLike


class HierarchySerializer(Protocol[T]):
    """Protocol for saving and loading hierarchy models."""

    async def save(self, hierarchy: HierarchyNode[T], path: PathLike) -> None:
        """Persist a hierarchy to disk."""

    async def load(self, path: PathLike) -> HierarchyNode[T]:
        """Load a hierarchy from disk."""


class JsonHierarchySerializer(HierarchySerializer[T]):
    """Serialize hierarchies as JSON using orjson."""

    async def save(self, hierarchy: HierarchyNode[T], path: PathLike) -> None:
        target_path = anyio.Path(path)
        await target_path.parent.mkdir(parents=True, exist_ok=True)
        await target_path.write_bytes(orjson.dumps(attrs.asdict(hierarchy), option=orjson.OPT_INDENT_2))

    async def load(self, path: PathLike) -> HierarchyNode[T]:
        data = orjson.loads(await anyio.Path(path).read_bytes())
        return HierarchyNode._from_dict(data)
