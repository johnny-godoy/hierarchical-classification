"""HuggingFace zero-shot classification integration.

Requires the ``huggingface`` optional dependency group::

    uv pip install hierarchical-classification[huggingface]

Uses a ``transformers`` zero-shot classification pipeline, so no labelled
training data is needed.  The child node *names* are used directly as
candidate labels.

Example::

    from src.integrations.huggingface import HuggingFaceZeroShotClassifier

    clf = HuggingFaceZeroShotClassifier()          # uses facebook/bart-large-mnli
    clf = HuggingFaceZeroShotClassifier("cross-encoder/nli-deberta-v3-small")
"""

import numpy as np
import numpy.typing as npt

from src.models import HierarchyNode, NodeClassifier

_DEFAULT_MODEL = "facebook/bart-large-mnli"


class HuggingFaceZeroShotClassifier(NodeClassifier):
    """Zero-shot node classifier backed by a HuggingFace ``transformers`` pipeline.

    The pipeline is instantiated lazily on the first call so that importing
    this module does not trigger a model download.

    Args:
        model: HuggingFace model name or local path for zero-shot classification.
            Defaults to ``"facebook/bart-large-mnli"``.
        **pipeline_kwargs: Additional keyword arguments forwarded to
            ``transformers.pipeline``.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, **pipeline_kwargs: object) -> None:
        self._model = model
        self._pipeline_kwargs = pipeline_kwargs
        self._pipe = None

    def _get_pipe(self) -> object:
        if self._pipe is None:
            from transformers import pipeline  # type: ignore[import-untyped]

            self._pipe = pipeline("zero-shot-classification", model=self._model, **self._pipeline_kwargs)
        return self._pipe

    def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
        """Classify *utterance* against each child of *node* using zero-shot NLI.

        Args:
            utterance: The text to classify.
            node: The current hierarchy node whose children are the candidates.

        Returns:
            A float32 array of probabilities aligned with ``node.children``.
        """
        child_names = [child.name for child in node.children]
        pipe = self._get_pipe()
        result: dict = pipe(utterance, candidate_labels=child_names)
        label_to_score: dict[str, float] = dict(zip(result["labels"], result["scores"]))
        return np.array([label_to_score[name] for name in child_names], dtype=np.float32)
