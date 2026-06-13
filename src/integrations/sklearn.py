"""Sklearn integration for hierarchical classification.

Requires the ``sklearn`` optional dependency group::

    uv pip install hierarchical-classification[sklearn]

Any sklearn-compatible classifier whose ``classes_`` attribute contains the
child node names can be plugged in directly.  A ``Pipeline`` that starts with
a text vectorizer (e.g. ``TfidfVectorizer``) and ends with a probabilistic
classifier (e.g. ``LogisticRegression``) is the most common setup.

Example::

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(X_train, y_train)

    from src.integrations.sklearn import SklearnNodeClassifier
    node_clf = SklearnNodeClassifier(pipe)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from src.models import HierarchyNode, NodeClassifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class SklearnNodeClassifier(NodeClassifier):
    """Wraps any sklearn-compatible classifier as a :class:`~src.models.NodeClassifier`.

    The wrapped classifier must expose ``predict_proba`` and a ``classes_``
    attribute (standard for all sklearn probabilistic classifiers).  Its
    ``classes_`` values must match the ``name`` attribute of the children of
    each :class:`~src.models.HierarchyNode` it will be asked to classify.

    Args:
        classifier: A fitted sklearn classifier or ``Pipeline`` that implements
            ``predict_proba`` and exposes ``classes_``.
    """

    def __init__(self, classifier: ClassifierMixin) -> None:
        self._classifier = classifier

    def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
        """Return per-child probabilities by selecting the relevant classes.

        Args:
            utterance: The text to classify.
            node: The current hierarchy node whose children are the candidates.

        Returns:
            A float32 array of probabilities aligned with ``node.children``.
        """
        child_names = [child.name for child in node.children]
        raw_proba: npt.NDArray[np.float64] = self._classifier.predict_proba([utterance])[0]
        class_to_prob: dict[str, float] = dict(zip(self._classifier.classes_, raw_proba))
        return np.array(
            [class_to_prob.get(name, 0.0) for name in child_names],
            dtype=np.float32,
        )
