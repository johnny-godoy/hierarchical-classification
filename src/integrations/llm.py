"""LLM integration for hierarchical classification via litellm.

Requires the ``llm`` optional dependency group::

    uv pip install hierarchical-classification[llm]

`litellm <https://github.com/BerriAI/litellm>`_ provides a unified interface
to 100+ LLM providers (OpenAI, Anthropic, Gemini, Ollama, …).  Set the
appropriate environment variable for your provider (e.g. ``OPENAI_API_KEY``)
before using this integration.

Example::

    from src.integrations.llm import LLMNodeClassifier

    clf = LLMNodeClassifier()  # defaults to gpt-4o-mini
    clf = LLMNodeClassifier("anthropic/claude-3-haiku-20240307")
    clf = LLMNodeClassifier("ollama/llama3")  # local model via Ollama
"""

import numpy as np
import numpy.typing as npt

from src.models import HierarchyNode, NodeClassifier

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a text classification assistant. "
    "Given a piece of text and a list of candidate categories, respond with "
    "exactly one category name from the list — nothing else."
)


class LLMNodeClassifier(NodeClassifier):
    """LLM-backed node classifier using litellm.

    The LLM is asked to pick the single best-matching child label for the
    given utterance.  The chosen label receives a probability of ``1.0``; all
    others receive ``0.0``.

    Args:
        model: Any model string accepted by ``litellm.completion`` (e.g.
            ``"gpt-4o-mini"``, ``"anthropic/claude-3-haiku-20240307"``,
            ``"ollama/llama3"``).  Defaults to ``"gpt-4o-mini"``.
        **completion_kwargs: Additional keyword arguments forwarded verbatim to
            ``litellm.completion`` (e.g. ``temperature``, ``api_base``).
    """

    def __init__(self, model: str = DEFAULT_MODEL, **completion_kwargs: object) -> None:
        self._model = model
        self._completion_kwargs = completion_kwargs

    def predict_proba(self, utterance: str, node: HierarchyNode) -> npt.NDArray[np.float32]:
        """Ask the LLM to select the best child category for *utterance*.

        Args:
            utterance: The text to classify.
            node: The current hierarchy node whose children are the candidates.

        Returns
        -------
            A float32 array where the chosen child has probability ``1.0`` and
            all others have ``0.0``.  If the model returns an unrecognised label
            the array is all zeros (no child is traversed).
        """
        import litellm  # type: ignore[import-untyped]  # noqa: PLC0415

        child_names = [child.name for child in node.children]
        user_content = f"Categories: {child_names}\n\nText: {utterance}\n\n Reply with the single best category name."
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            **self._completion_kwargs,
        )
        chosen: str = response.choices[0].message.content.strip()
        proba = np.zeros(len(child_names), dtype=np.float32)
        if chosen in child_names:
            proba[child_names.index(chosen)] = 1.0
        return proba
