"""Integration tests: end-to-end HierarchicalClassifier with each integration type.

External library calls (transformers pipeline, litellm.completion) are mocked so
these tests remain fast and offline, while still exercising the full
HierarchicalClassifier → ScoringStrategy → NodeClassifier → predict_proba → classify
chain without any stubbing of internal code.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.classifier import HierarchicalClassifier
from src.integrations.huggingface import HuggingFaceZeroShotClassifier
from src.integrations.llm import LLMNodeClassifier
from src.integrations.sklearn import SklearnNodeClassifier
from src.models import HierarchyNode

# ---------------------------------------------------------------------------
# Shared hierarchy
# ---------------------------------------------------------------------------


@pytest.fixture
def animal_hierarchy() -> HierarchyNode:
    """Root → {animal, vehicle} → leaves."""
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
# Sklearn integration
# ---------------------------------------------------------------------------


class TestSklearnIntegration:
    def test_full_classify_with_sklearn(self, animal_hierarchy: HierarchyNode) -> None:
        # Classifier whose classes_ covers all node names and gives high probability
        # to "animal" and "cat" so the traversal ends at "cat".
        all_names = ["root", "animal", "vehicle", "cat", "dog", "car", "bike"]

        class FullClf:
            classes_ = np.array(all_names)

            def predict_proba(self, _: str) -> np.ndarray:
                row = np.zeros(len(self.classes_))
                for i, name in enumerate(self.classes_):
                    if name in {"animal", "cat"}:
                        row[i] = 0.9
                    else:
                        row[i] = 0.02
                return np.array([row])

        node_clf = SklearnNodeClassifier(FullClf())
        hc = HierarchicalClassifier.from_classifier(node_clf, animal_hierarchy)
        result = hc.classify("a cat is an animal")
        assert result == "cat"

    def test_full_classify_vehicle_branch(self, animal_hierarchy: HierarchyNode) -> None:
        all_names = ["root", "animal", "vehicle", "cat", "dog", "car", "bike"]

        class VehicleClf:
            classes_ = np.array(all_names)

            def predict_proba(self, _: str) -> np.ndarray:
                row = np.zeros(len(self.classes_))
                for i, name in enumerate(self.classes_):
                    if name in {"vehicle", "bike"}:
                        row[i] = 0.9
                    else:
                        row[i] = 0.02
                return np.array([row])

        node_clf = SklearnNodeClassifier(VehicleClf())
        hc = HierarchicalClassifier.from_classifier(node_clf, animal_hierarchy)
        result = hc.classify("riding my bike")
        assert result == "bike"


# ---------------------------------------------------------------------------
# HuggingFace integration
# ---------------------------------------------------------------------------


def _hf_pipeline_factory(routing: dict[str, str]) -> callable:
    """Return a callable that acts like a zero-shot pipeline, routing by utterance."""

    def _pipeline(utterance: str, candidate_labels: list[str]) -> dict[str, list[float]]:
        chosen = routing.get(utterance, candidate_labels[0])
        n = len(candidate_labels)
        scores = {lbl: 0.05 / max(n - 1, 1) for lbl in candidate_labels}
        if chosen in scores:
            scores[chosen] = 0.95
        labels = candidate_labels
        return {"labels": labels, "scores": [scores[label] for label in labels]}

    return _pipeline


class TestHuggingFaceIntegration:
    def test_full_classify_animal(self, animal_hierarchy: HierarchyNode) -> None:
        # "meow" at root → animal; "meow" at animal node → first child (cat)
        routing = {"meow": "animal"}
        mock_pipe = _hf_pipeline_factory(routing)

        clf = HuggingFaceZeroShotClassifier()
        clf._pipe = mock_pipe

        hc = HierarchicalClassifier.from_classifier(clf, animal_hierarchy)
        result = hc.classify("meow")
        assert result == "cat"

    def test_full_classify_vehicle(self, animal_hierarchy: HierarchyNode) -> None:
        routing = {"vroom": "vehicle", "car": "car"}
        mock_pipe = _hf_pipeline_factory(routing)

        clf = HuggingFaceZeroShotClassifier()
        clf._pipe = mock_pipe

        hc = HierarchicalClassifier.from_classifier(clf, animal_hierarchy)
        result = hc.classify("vroom")
        assert result == "car"

    def test_pipeline_instantiated_via_transformers(self) -> None:
        node = HierarchyNode(name="root", children=[HierarchyNode(name="cat")])
        clf = HuggingFaceZeroShotClassifier("my-model")

        mock_transformers = MagicMock()
        pipe_instance = MagicMock(return_value={"labels": ["cat"], "scores": [1.0]})
        mock_transformers.pipeline.return_value = pipe_instance

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            clf.predict_proba("meow", node)

        mock_transformers.pipeline.assert_called_once_with("zero-shot-classification", model="my-model")


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------


def _make_litellm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_litellm() -> MagicMock:
    mock = MagicMock()
    with patch.dict(sys.modules, {"litellm": mock}):
        yield mock


class TestLLMIntegration:
    def test_full_classify_routes_to_correct_leaf(
        self,
        mock_litellm: MagicMock,
        animal_hierarchy: HierarchyNode,
    ) -> None:
        responses = iter(["animal", "dog"])
        mock_litellm.completion.side_effect = lambda **_: _make_litellm_response(next(responses))

        clf = LLMNodeClassifier()
        hc = HierarchicalClassifier.from_classifier(clf, animal_hierarchy)
        result = hc.classify("the dog barked")

        assert result == "dog"

    def test_full_classify_with_vehicle(self, mock_litellm: MagicMock, animal_hierarchy: HierarchyNode) -> None:
        responses = iter(["vehicle", "bike"])
        mock_litellm.completion.side_effect = lambda **_: _make_litellm_response(next(responses))

        clf = LLMNodeClassifier("ollama/llama3")
        hc = HierarchicalClassifier.from_classifier(clf, animal_hierarchy)
        result = hc.classify("riding my bike")

        assert result == "bike"

    def test_unrecognised_llm_output_raises(self, mock_litellm: MagicMock, animal_hierarchy: HierarchyNode) -> None:
        """If the LLM returns garbage at every level no leaf is reachable."""
        mock_litellm.completion.return_value = _make_litellm_response("GARBAGE")

        clf = LLMNodeClassifier()
        hc = HierarchicalClassifier.from_classifier(clf, animal_hierarchy)
        with pytest.raises(ValueError, match="No leaf node found"):
            hc.classify("nonsense")
