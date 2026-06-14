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

    @staticmethod
    async def save(hierarchy: HierarchyNode[T], path: PathLike) -> None:
        """Persist a hierarchy to disk."""

    @staticmethod
    async def load(path: PathLike) -> HierarchyNode[T]:
        """Load a hierarchy from disk."""


class JsonHierarchySerializer(HierarchySerializer[T]):
    """Serialize hierarchies as JSON using orjson."""

    @staticmethod
    async def save(hierarchy: HierarchyNode[T], path: PathLike) -> None:
        """Save a hierarchy to disk as JSON."""
        target_path = anyio.Path(path)
        await target_path.parent.mkdir(parents=True, exist_ok=True)
        await target_path.write_bytes(orjson.dumps(attrs.asdict(hierarchy), option=orjson.OPT_INDENT_2))

    @staticmethod
    async def load(path: PathLike) -> HierarchyNode[T]:
        """Load a hierarchy from disk as JSON.

        Parameters
        ----------
        path: PathLike
            The file path to load the hierarchy from.

        Returns
        -------
            A HierarchyNode instance reconstructed from the JSON file.
        """
        data = orjson.loads(await anyio.Path(path).read_bytes())
        return HierarchyNode.from__dict(data)
