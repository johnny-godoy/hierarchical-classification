"""Unit tests for src/integrations/huggingface.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.integrations.huggingface import HuggingFaceZeroShotClassifier, _DEFAULT_MODEL
from src.models import HierarchyNode


def _make_hf_result(labels: list[str], scores: list[float]) -> dict:
    """Build the dict that a real HuggingFace zero-shot pipeline returns."""
    return {"labels": labels, "scores": scores}


@pytest.fixture()
def node_animals() -> HierarchyNode:
    return HierarchyNode(
        name="root",
        children=[HierarchyNode(name="cat"), HierarchyNode(name="dog"), HierarchyNode(name="bird")],
    )


class TestHuggingFaceZeroShotClassifier:
    def _make_clf_with_mock_pipe(self, node: HierarchyNode, scores_by_label: dict[str, float]):
        clf = HuggingFaceZeroShotClassifier()
        mock_pipe = MagicMock()
        labels = [c.name for c in node.children]
        mock_pipe.return_value = _make_hf_result(labels, [scores_by_label[l] for l in labels])
        clf._pipe = mock_pipe
        return clf

    def test_returns_probabilities_aligned_with_children(self, node_animals) -> None:
        clf = self._make_clf_with_mock_pipe(node_animals, {"cat": 0.6, "dog": 0.3, "bird": 0.1})
        proba = clf.predict_proba("meow", node_animals)
        assert len(proba) == 3
        np.testing.assert_allclose(proba, [0.6, 0.3, 0.1], atol=1e-6)

    def test_dtype_is_float32(self, node_animals) -> None:
        clf = self._make_clf_with_mock_pipe(node_animals, {"cat": 0.5, "dog": 0.3, "bird": 0.2})
        proba = clf.predict_proba("test", node_animals)
        assert proba.dtype == np.float32

    def test_pipe_called_with_correct_args(self, node_animals) -> None:
        clf = HuggingFaceZeroShotClassifier()
        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_hf_result(["cat", "dog", "bird"], [0.5, 0.3, 0.2])
        clf._pipe = mock_pipe

        clf.predict_proba("hello kitty", node_animals)

        mock_pipe.assert_called_once_with(
            "hello kitty",
            candidate_labels=["cat", "dog", "bird"],
        )

    def test_pipeline_loaded_lazily(self) -> None:
        clf = HuggingFaceZeroShotClassifier()
        assert clf._pipe is None  # not loaded yet

    def test_pipeline_cached_after_first_call(self, node_animals) -> None:
        clf = HuggingFaceZeroShotClassifier()
        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_hf_result(["cat", "dog", "bird"], [0.5, 0.3, 0.2])

        with patch("src.integrations.huggingface.HuggingFaceZeroShotClassifier._get_pipe", return_value=mock_pipe):
            clf.predict_proba("test", node_animals)
            clf.predict_proba("test2", node_animals)

        assert mock_pipe.call_count == 2  # same pipe object called twice

    def test_default_model_used(self) -> None:
        clf = HuggingFaceZeroShotClassifier()
        assert clf._model == _DEFAULT_MODEL

    def test_custom_model_stored(self) -> None:
        clf = HuggingFaceZeroShotClassifier("cross-encoder/nli-deberta-v3-small")
        assert clf._model == "cross-encoder/nli-deberta-v3-small"

    def test_pipeline_kwargs_forwarded(self) -> None:
        import sys
        from unittest.mock import patch

        mock_transformers = MagicMock()
        pipe_instance = MagicMock(return_value={"labels": ["cat"], "scores": [1.0]})
        mock_transformers.pipeline.return_value = pipe_instance

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            clf = HuggingFaceZeroShotClassifier(device="cpu")
            node = HierarchyNode(name="root", children=[HierarchyNode(name="cat")])
            clf.predict_proba("test", node)

        mock_transformers.pipeline.assert_called_once_with(
            "zero-shot-classification",
            model=_DEFAULT_MODEL,
            device="cpu",
        )

    def test_order_preserved_when_hf_reorders_labels(self, node_animals) -> None:
        """HuggingFace may return labels in a different order than supplied."""
        clf = HuggingFaceZeroShotClassifier()
        mock_pipe = MagicMock()
        # HF returned labels in reversed order
        mock_pipe.return_value = _make_hf_result(
            ["bird", "dog", "cat"], [0.1, 0.3, 0.6]
        )
        clf._pipe = mock_pipe

        proba = clf.predict_proba("meow", node_animals)
        # Must be aligned with node.children order: cat, dog, bird
        np.testing.assert_allclose(proba, [0.6, 0.3, 0.1], atol=1e-6)
