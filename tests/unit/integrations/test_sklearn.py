"""Unit tests for src/integrations/sklearn.py."""

import numpy as np
import pytest

from src.integrations.sklearn import SklearnNodeClassifier
from src.models import HierarchyNode


class _FakeSklearnClf:
    """Minimal sklearn-compatible classifier stub (no fitting needed)."""

    def __init__(self, classes_: list[str], proba_map: dict[str, float]) -> None:
        self.classes_ = np.array(classes_)
        self._proba_map = proba_map

    def predict_proba(self, _: str) -> np.ndarray:
        row = [self._proba_map.get(c, 0.0) for c in self.classes_]
        return np.array([row])


@pytest.fixture
def node_with_cat_dog() -> HierarchyNode:
    return HierarchyNode(
        name="root",
        children=[HierarchyNode(name="cat"), HierarchyNode(name="dog")],
    )


@pytest.fixture
def sklearn_clf_cat_dog(
    node_with_cat_dog: HierarchyNode,  # noqa: ARG001
) -> SklearnNodeClassifier:
    inner = _FakeSklearnClf(["cat", "dog"], {"cat": 0.7, "dog": 0.3})
    return SklearnNodeClassifier(inner)


class TestSklearnNodeClassifier:
    def test_returns_correct_probabilities(
        self,
        sklearn_clf_cat_dog: SklearnNodeClassifier,
        node_with_cat_dog: HierarchyNode,
    ) -> None:
        proba = sklearn_clf_cat_dog.predict_proba("a cat", node_with_cat_dog)
        assert proba.dtype == np.float32
        np.testing.assert_allclose(proba, [0.7, 0.3], atol=1e-6)

    def test_length_matches_children(
        self,
        sklearn_clf_cat_dog: SklearnNodeClassifier,
        node_with_cat_dog: HierarchyNode,
    ) -> None:
        proba = sklearn_clf_cat_dog.predict_proba("test", node_with_cat_dog)
        assert len(proba) == len(node_with_cat_dog.children)

    def test_unknown_child_gets_zero(self) -> None:
        """Child name not in classifier classes_ → probability 0."""
        node = HierarchyNode(
            name="root",
            children=[HierarchyNode(name="unknown_class"), HierarchyNode(name="cat")],
        )
        inner = _FakeSklearnClf(["cat", "dog"], {"cat": 0.8, "dog": 0.2})
        clf = SklearnNodeClassifier(inner)
        proba = clf.predict_proba("cat", node)
        assert proba[0] == pytest.approx(0.0)  # unknown_class → 0
        assert proba[1] == pytest.approx(0.8)  # cat → 0.8

    def test_probabilities_are_float32(
        self,
        sklearn_clf_cat_dog: SklearnNodeClassifier,
        node_with_cat_dog: HierarchyNode,
    ) -> None:
        proba = sklearn_clf_cat_dog.predict_proba("x", node_with_cat_dog)
        assert proba.dtype == np.float32

    def test_order_matches_children_not_classes(self) -> None:
        """Returned array must be aligned with node.children, not classes_."""
        node = HierarchyNode(
            name="root",
            children=[HierarchyNode(name="dog"), HierarchyNode(name="cat")],
        )
        # classes_ in alphabetical order (typical sklearn behaviour)
        inner = _FakeSklearnClf(["cat", "dog"], {"cat": 0.3, "dog": 0.7})
        clf = SklearnNodeClassifier(inner)
        proba = clf.predict_proba("woof", node)
        # children order: dog, cat → expected [0.7, 0.3]
        np.testing.assert_allclose(proba, [0.7, 0.3], atol=1e-6)

    def test_single_child(self) -> None:
        node = HierarchyNode(name="root", children=[HierarchyNode(name="only")])
        inner = _FakeSklearnClf(["only"], {"only": 1.0})
        clf = SklearnNodeClassifier(inner)
        proba = clf.predict_proba("test", node)
        assert len(proba) == 1
        assert proba[0] == pytest.approx(1.0)
