"""Unit tests for src/integrations/llm.py."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.integrations.llm import DEFAULT_MODEL, SYSTEM_PROMPT, LLMNodeClassifier
from src.models import HierarchyNode


def _make_litellm_response(content: str) -> MagicMock:
    """Build a minimal litellm-style response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_litellm() -> MagicMock:
    """Fixture that injects a mock litellm module into sys.modules."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"litellm": mock}):
        yield mock


@pytest.fixture
def node_animals() -> HierarchyNode:
    return HierarchyNode(
        name="root",
        children=[HierarchyNode(name="cat"), HierarchyNode(name="dog"), HierarchyNode(name="bird")],
    )


class TestLLMNodeClassifier:
    def test_correct_label_gets_probability_one(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("dog")
        clf = LLMNodeClassifier()
        proba = clf.predict_proba("woof woof", node_animals)
        np.testing.assert_allclose(proba, [0.0, 1.0, 0.0], atol=1e-6)

    def test_dtype_is_float32(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("cat")
        clf = LLMNodeClassifier()
        proba = clf.predict_proba("meow", node_animals)
        assert proba.dtype == np.float32

    def test_unrecognised_label_returns_all_zeros(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("fish")
        clf = LLMNodeClassifier()
        proba = clf.predict_proba("blub", node_animals)
        np.testing.assert_allclose(proba, [0.0, 0.0, 0.0], atol=1e-6)

    def test_whitespace_stripped_from_response(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("  bird  \n")
        clf = LLMNodeClassifier()
        proba = clf.predict_proba("tweet", node_animals)
        np.testing.assert_allclose(proba, [0.0, 0.0, 1.0], atol=1e-6)

    def test_length_equals_number_of_children(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("cat")
        clf = LLMNodeClassifier()
        proba = clf.predict_proba("test", node_animals)
        assert len(proba) == len(node_animals.children)

    def test_DEFAULT_MODEL_used(self) -> None:  # noqa: N802
        clf = LLMNodeClassifier()
        assert clf._model == DEFAULT_MODEL

    def test_custom_model_stored(self) -> None:
        clf = LLMNodeClassifier("anthropic/claude-3-haiku-20240307")
        assert clf._model == "anthropic/claude-3-haiku-20240307"

    def test_completion_kwargs_forwarded(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("cat")
        clf = LLMNodeClassifier(temperature=0.0)
        clf.predict_proba("meow", node_animals)
        _, call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.get("temperature") == 0.0

    def test_correct_messages_sent_to_litellm(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("cat")
        clf = LLMNodeClassifier()
        clf.predict_proba("meow", node_animals)
        _, call_kwargs = mock_litellm.completion.call_args
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "meow" in messages[1]["content"]
        assert "cat" in messages[1]["content"]

    def test_model_name_forwarded_to_litellm(self, mock_litellm: MagicMock, node_animals: HierarchyNode) -> None:
        mock_litellm.completion.return_value = _make_litellm_response("cat")
        clf = LLMNodeClassifier("ollama/llama3")
        clf.predict_proba("meow", node_animals)
        _, call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs["model"] == "ollama/llama3"

    def test_single_child(self, mock_litellm: MagicMock) -> None:
        node = HierarchyNode(name="root", children=[HierarchyNode(name="only")])
        mock_litellm.completion.return_value = _make_litellm_response("only")
        clf = LLMNodeClassifier()
        proba = clf.predict_proba("test", node)
        assert proba[0] == pytest.approx(1.0)
