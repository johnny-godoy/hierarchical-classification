import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    def _find_project_root() -> None:
        """Add the project root (the directory containing ``src/``) to sys.path."""
        cwd = Path.cwd()
        for candidate in [cwd, cwd.parent]:
            if (candidate / "src").is_dir():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return

    _find_project_root()

    import numpy as np

    from src.classifier import HierarchicalClassifier
    from src.models import HierarchyNode

    return HierarchicalClassifier, HierarchyNode, np


@app.cell
def _(mo):
    mo.md(r"""
    # Hierarchical Classification — Custom Classifier

    This notebook walks through building a **custom `NodeClassifier`** and wiring it
    into `HierarchicalClassifier`.

    A hierarchical classifier traverses a tree of categories, scoring each node with
    your classifier, and returns the best-matching **leaf** for a given text.

    ## How it works

    ```
    utterance ──► HierarchicalClassifier
                          │
               ┌──────────▼──────────┐
               │   NodeClassifier    │  ◄── your custom logic lives here
               │  .predict_proba()   │
               └──────────┬──────────┘
                          │  probabilities per child
               ┌──────────▼──────────┐
               │   Hierarchy tree    │
               │   (HierarchyNode)   │
               └──────────┬──────────┘
                          │
                       leaf node  ──► result
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — Define the hierarchy

    A `HierarchyNode` is a tree node with a name and optional children.
    Leaf nodes (no children) are the final categories the classifier can return.

    Below is a **news article** taxonomy with three top-level topics and nine leaf categories.
    """)
    return


@app.cell
def _(HierarchyNode):
    hierarchy = HierarchyNode(
        name="root",
        children=[
            HierarchyNode(
                name="Sports",
                children=[
                    HierarchyNode(name="Football"),
                    HierarchyNode(name="Basketball"),
                    HierarchyNode(name="Tennis"),
                ],
            ),
            HierarchyNode(
                name="Technology",
                children=[
                    HierarchyNode(name="Artificial Intelligence"),
                    HierarchyNode(name="Programming"),
                    HierarchyNode(name="Hardware"),
                ],
            ),
            HierarchyNode(
                name="Health",
                children=[
                    HierarchyNode(name="Nutrition"),
                    HierarchyNode(name="Exercise"),
                    HierarchyNode(name="Mental Health"),
                ],
            ),
        ],
    )
    return (hierarchy,)


@app.cell
def _(hierarchy, mo):
    def _tree_md(node, depth: int = 0) -> list[str]:
        icon = "📁" if node.children else "📄"
        indent = "      " * depth
        lines = [f"{indent}{icon} **{node.name}**"]
        for child in node.children:
            lines.extend(_tree_md(child, depth + 1))
        return lines

    tree = "\n\n".join(_tree_md(hierarchy))
    mo.accordion({"🗂️ Show hierarchy tree": mo.md(tree)})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2 — Implement a custom `NodeClassifier`

    The `NodeClassifier` protocol requires one method:

    ```python
    def predict_proba(self, utterance: str, node: HierarchyNode) -> np.ndarray:
        ...
    ```

    It must return a **probability array** — one entry per child of `node`, summing
    to 1.  Any approach works: keyword matching, ML model, rules, or LLM.

    Inheriting from `NodeClassifier` gives you the default `neg_log_proba`
    implementation for free, so only `predict_proba` needs to be overridden
    (the same pattern used by all built-in integrations).

    Here we build a **keyword-matching** classifier that scores each child by counting
    matching keywords in the utterance.
    """)
    return


@app.cell
def _(np):
    from src.models import HierarchyNode as HierarchyNodeModel
    from src.models import NodeClassifier as NodeClassifierModel

    keywords: dict[str, list[str]] = {
        # Leaf nodes
        "Football": ["football", "soccer", "goal", "match", "striker", "penalty", "pitch", "referee", "offside"],
        "Basketball": ["basketball", "nba", "hoop", "dunk", "three-pointer", "court", "rebound", "layup"],
        "Tennis": ["tennis", "serve", "ace", "wimbledon", "racket", "grand slam", "rally", "forehand"],
        "Artificial Intelligence": [
            "ai",
            "machine learning",
            "neural",
            "llm",
            "gpt",
            "model",
            "deep learning",
            "training",
            "dataset",
        ],
        "Programming": ["code", "python", "software", "developer", "api", "bug", "framework", "github", "pull request"],
        "Hardware": ["cpu", "gpu", "chip", "processor", "motherboard", "ram", "server", "ssd", "bandwidth"],
        "Nutrition": ["diet", "vitamins", "protein", "calories", "food", "nutrients", "eating", "weight", "meal"],
        "Exercise": ["workout", "gym", "running", "fitness", "cardio", "training", "muscles", "yoga", "jogging"],
        "Mental Health": ["anxiety", "depression", "therapy", "stress", "mindfulness", "wellbeing", "mood", "mental"],
        # Internal nodes (used at the top level of the tree)
        "Sports": ["sport", "athlete", "game", "team", "score", "championship", "player", "league", "win"],
        "Technology": ["technology", "tech", "digital", "computer", "software", "internet", "device", "innovation"],
        "Health": ["health", "medical", "wellness", "body", "disease", "doctor", "clinic", "patient", "treatment"],
    }

    class KeywordNodeClassifier(NodeClassifierModel):
        """
        A keyword-based :class:`~src.models.NodeClassifier` for demonstration purposes.

        Subclassing :class:`~src.models.NodeClassifier` inherits the default
        ``neg_log_proba`` implementation, so only ``predict_proba`` needs to be
        overridden.

        Each child category has a set of associated keywords.  The probability of a
        child is proportional to the number of its keywords found in the utterance,
        with a small smoothing term so that no child is completely excluded.
        """

        def predict_proba(self, utterance: str, node: HierarchyNodeModel) -> np.ndarray:  # noqa: PLR6301
            text = utterance.lower()
            scores = np.array(
                [sum(1.0 for kw in keywords.get(child.name, []) if kw in text) + 0.1 for child in node.children],
                dtype=np.float32,
            )
            return scores / scores.sum()

    return keywords, KeywordNodeClassifier


@app.cell
def _(keywords: dict[str, list[str]], mo):
    rows = [{"Category": cat, "Keywords": ", ".join(kws)} for cat, kws in keywords.items()]
    mo.accordion(
        {"🔑 Show keyword dictionary": mo.ui.table(rows, selection=None)},
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3 — Try it live!

    Pick a preset or type your own text to watch the classifier traverse the hierarchy
    in real time.
    """)
    return


@app.cell
def _(mo):
    preset_picker = mo.ui.dropdown(
        options={
            "AI article": "The new transformer model achieves state-of-the-art results on language benchmarks.",
            "Football match": "Real Madrid scored three goals in the second half to claim the title.",
            "Python bug": "There is a null pointer exception in the API endpoint that crashes the server.",
            "Nutrition advice": "Eating enough protein and vitamins is key to maintaining a healthy weight.",
            "GPU hardware": "The new GPU chipset doubles memory bandwidth compared to the previous generation.",
            "Mental wellbeing": "Daily mindfulness practice has been shown to reduce anxiety and stress.",
            "Tennis final": "She won the Wimbledon final with a decisive ace on match point.",
            "Custom (edit below)": "",
        },
        value="AI article",
        label="Pick a preset example",
    )
    return (preset_picker,)


@app.cell
def _(mo, preset_picker):
    initial = (
        preset_picker.value or "The new transformer model achieves state-of-the-art results on language benchmarks."
    )
    utterance_input = mo.ui.text_area(
        value=initial,
        label="✏️ Text to classify",
        full_width=True,
        rows=3,
    )
    return (utterance_input,)


@app.cell
def _(
    HierarchicalClassifier,
    KeywordNodeClassifier,
    hierarchy,
    mo,
    utterance_input,
):
    text = utterance_input.value.strip()
    if text:
        clf = HierarchicalClassifier.from_classifier(
            node_classifier=KeywordNodeClassifier(),
            hierarchy=hierarchy,
        )
        result = clf.classify(text)
        mo.callout(
            mo.md(f"🏷️  **Predicted category:** {result}"),
            kind="success",
        )
    else:
        mo.callout(mo.md("_Enter some text above to see the predicted category._"), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4 — Probability breakdown

    The table below shows the raw probability assigned to every node at each level
    of the hierarchy, so you can see exactly how the traversal scores each branch.
    """)
    return


@app.cell
def _(KEYWORDS: dict[str, list[str]], hierarchy, mo, np, utterance_input):
    text = utterance_input.value.strip().lower()

    def _score(node_name: str) -> float:
        kws = KEYWORDS.get(node_name, [])
        return sum(1.0 for kw in kws if kw in text) + 0.1

    rows = []
    for level_node in [hierarchy, *hierarchy.children]:
        if not level_node.children:
            continue
        raw = np.array([_score(c.name) for c in level_node.children], dtype=np.float32)
        proba = raw / raw.sum()
        for child, p in zip(level_node.children, proba, strict=True):
            rows.append(
                {
                    "Parent": level_node.name,
                    "Category": child.name,
                    "Probability": f"{p:.3f}",
                    "Relative weight": "█" * max(1, round(p * 20)),
                },
            )

    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next steps

    - **Replace the keyword classifier** with an ML model — see the **sklearn
      integration** notebook for a drop-in example using `TF-IDF + LogisticRegression`.
    - **Persist the hierarchy** with `await hierarchy.save("hierarchy.json")` and reload
      it with `await HierarchyNode.load("hierarchy.json")`.
    - **Attach training examples** to nodes via `HierarchyNode.examples` to keep
      your labelled data alongside the hierarchy definition.
    - **Zero-shot classification** — try `HuggingFaceZeroShotClassifier` from
      `src.integrations.huggingface` when you have no labelled data at all.
    """)
    return


if __name__ == "__main__":
    app.run()
