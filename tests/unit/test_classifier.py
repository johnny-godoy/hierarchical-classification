"""Unit tests for src/classifier.py."""

import numpy as np
import numpy.typing as npt
import pytest

from src.classifier import HierarchicalClassifier
from src.models import HierarchyNode, NodeClassifier
from tests.conftest import DeterministicClassifier


class TestHierarchicalClassifierClassify:
    def test_flat_hierarchy_picks_first_child(self, flat_hierarchy: HierarchyNode, first_child_classifier) -> None:
        hc = HierarchicalClassifier(classifier=first_child_classifier, hierarchy=flat_hierarchy)
        result = hc.classify("any text")
        assert result == "cat"

    def test_flat_hierarchy_picks_second_child(self, flat_hierarchy: HierarchyNode) -> None:
        clf = DeterministicClassifier(target_index=1)
        hc = HierarchicalClassifier(classifier=clf, hierarchy=flat_hierarchy)
        result = hc.classify("any text")
        assert result == "dog"

    def test_flat_hierarchy_picks_third_child(self, flat_hierarchy: HierarchyNode) -> None:
        clf = DeterministicClassifier(target_index=2)
        hc = HierarchicalClassifier(classifier=clf, hierarchy=flat_hierarchy)
        result = hc.classify("any text")
        assert result == "bird"

    def test_deep_hierarchy_traverses_to_leaf(self, deep_hierarchy: HierarchyNode, first_child_classifier) -> None:
        # first child of root = "animal", first child of "animal" = "cat"
        hc = HierarchicalClassifier(classifier=first_child_classifier, hierarchy=deep_hierarchy)
        result = hc.classify("meow")
        assert result == "cat"

    def test_deep_hierarchy_second_branch(self, deep_hierarchy: HierarchyNode) -> None:
        # always picks last child: root's last = "vehicle", vehicle's last = "bike"
        class LastChildClassifier(NodeClassifier):
            def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
                n = len(node.children)
                proba = np.zeros(n, dtype=np.float32)
                if n:
                    proba[-1] = 1.0
                return proba

        hc = HierarchicalClassifier(classifier=LastChildClassifier(), hierarchy=deep_hierarchy)
        result = hc.classify("ride a bike")
        assert result == "bike"

    def test_raises_when_no_leaf_reachable(self) -> None:
        root = HierarchyNode(name="root", children=[HierarchyNode(name="child")])

        class NeverClassifier(NodeClassifier):
            def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
                return np.zeros(len(node.children), dtype=np.float32)

        hc = HierarchicalClassifier(classifier=NeverClassifier(), hierarchy=root)
        with pytest.raises(ValueError, match="No leaf node found"):
            hc.classify("unreachable")

    def test_single_leaf_as_root_returns_its_name(self) -> None:
        """Root that is itself a leaf (no children)."""
        root = HierarchyNode(name="only_leaf")
        hc = HierarchicalClassifier(classifier=DeterministicClassifier(), hierarchy=root)
        # The root is a leaf, so it should be returned immediately.
        result = hc.classify("anything")
        assert result == "only_leaf"

    def test_utterance_passed_through(self, flat_hierarchy: HierarchyNode) -> None:
        """Confirm the utterance is forwarded to predict_proba."""
        received: list[str] = []

        class RecordingClassifier(NodeClassifier):
            def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
                received.append(utterance)
                proba = np.zeros(len(node.children), dtype=np.float32)
                if node.children:
                    proba[0] = 1.0
                return proba

        hc = HierarchicalClassifier(classifier=RecordingClassifier(), hierarchy=flat_hierarchy)
        hc.classify("hello world")
        assert "hello world" in received

    def test_best_path_selected_over_suboptimal(self, deep_hierarchy: HierarchyNode) -> None:
        """Classifier that routes animal→cat with high prob but vehicle→car with low prob."""

        class WeightedClassifier(NodeClassifier):
            def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
                child_names = [c.name for c in node.children]
                if node.name == "root":
                    # strongly prefer animal
                    return np.array([0.9, 0.1], dtype=np.float32)
                if node.name == "animal":
                    return np.array([0.8, 0.2], dtype=np.float32)  # cat, dog
                if node.name == "vehicle":
                    return np.array([0.6, 0.4], dtype=np.float32)  # car, bike
                return np.ones(len(child_names), dtype=np.float32) / len(child_names)

        hc = HierarchicalClassifier(classifier=WeightedClassifier(), hierarchy=deep_hierarchy)
        result = hc.classify("meow")
        assert result == "cat"
