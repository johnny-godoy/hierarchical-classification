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

    from src.classifier import HierarchicalClassifier
    from src.integrations.sklearn import SklearnNodeClassifier
    from src.models import HierarchyNode

    return HierarchicalClassifier, HierarchyNode, SklearnNodeClassifier


@app.cell
def _(mo):
    mo.md(r"""
    # Hierarchical Classification — Sklearn Integration

    This notebook demonstrates the **sklearn integration** bundled with this library.

    `SklearnNodeClassifier` wraps any scikit-learn probabilistic classifier — typically
    a `Pipeline` with a text vectoriser and a classifier that implements
    `predict_proba` — and adapts it to the `NodeClassifier` protocol.

    ## How it works

    The **same** fitted sklearn model is reused at every node in the hierarchy.
    When `HierarchicalClassifier` visits a node it calls `predict_proba` for the
    **child labels** of that node, then follows the highest-probability path down to
    a leaf.

    ```
    utterance
        │
        ▼
    SklearnNodeClassifier.predict_proba(utterance, node)
        │  selects probabilities for node.children from the fitted pipeline
        ▼
    HierarchicalClassifier (best-first traversal)
        │
        ▼
      leaf node  ──►  result
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — Training data

    Because **the same** fitted sklearn model is reused at every tree node, its
    `classes_` must cover the children of **every** node it will be asked to score.
    Our two-level hierarchy has 3 top-level nodes (`Sports`, `Technology`, `Health`)
    and 9 leaf nodes, so the model must be trained to predict all 12 labels.

    The table below shows one row per training example. Top-level examples teach the
    model to distinguish broad topics; leaf examples give fine-grained signal.
    """)
    return


@app.cell
def _():
    training_data: list[tuple[str, str]] = [
        # ── Top-level: Sports ──────────────────────────────────────────────
        ("This week's top sports headlines from around the world.", "Sports"),
        ("Athletes from 50 nations compete in the championship.", "Sports"),
        ("The sports industry has seen record investment this year.", "Sports"),
        ("Young players are being scouted by professional leagues worldwide.", "Sports"),
        # ── Top-level: Technology ──────────────────────────────────────────
        ("Tech companies are announcing major product launches this quarter.", "Technology"),
        ("Innovation in digital devices is accelerating faster than ever.", "Technology"),
        ("The technology sector posted record revenue last year.", "Technology"),
        ("Advances in computing are transforming every industry.", "Technology"),
        # ── Top-level: Health ──────────────────────────────────────────────
        ("Public health officials issued new guidelines today.", "Health"),
        ("Medical researchers published findings on a new treatment.", "Health"),
        ("Wellness trends are reshaping how people approach their daily routines.", "Health"),
        ("Healthcare providers are adopting new digital tools to improve care.", "Health"),
        # ── Football ───────────────────────────────────────────────────────
        ("The striker scored a brilliant goal in the 89th minute.", "Football"),
        ("The referee gave a red card after a dangerous tackle.", "Football"),
        ("The team advanced to the semi-finals after a penalty shootout.", "Football"),
        ("The new football stadium will host 80,000 fans.", "Football"),
        ("The midfielder completed 95 percent of his passes in the match.", "Football"),
        # ── Basketball ─────────────────────────────────────────────────────
        ("LeBron James hit a three-pointer to tie the game.", "Basketball"),
        ("The NBA finals drew record television viewership.", "Basketball"),
        ("The point guard dribbled past two defenders for a slam dunk.", "Basketball"),
        ("The basketball team won their eighth championship title.", "Basketball"),
        ("The rookie's rebounding changed the momentum of the game.", "Basketball"),
        # ── Tennis ─────────────────────────────────────────────────────────
        ("She won the Wimbledon title with a dominant serve.", "Tennis"),
        ("The grand slam final lasted five sets before a winner emerged.", "Tennis"),
        ("His forehand was too powerful for his opponent at the net.", "Tennis"),
        ("The young tennis prodigy upset the world number one.", "Tennis"),
        ("The ace on match point sealed a historic victory.", "Tennis"),
        # ── Artificial Intelligence ────────────────────────────────────────
        ("The new language model achieves state-of-the-art results on benchmarks.", "Artificial Intelligence"),
        ("Researchers trained a neural network on one trillion tokens.", "Artificial Intelligence"),
        ("The deep learning model outperforms human experts in medical imaging.", "Artificial Intelligence"),
        ("GPT-based systems are changing how developers write code.", "Artificial Intelligence"),
        ("The AI system learned to play chess at grandmaster level.", "Artificial Intelligence"),
        # ── Programming ────────────────────────────────────────────────────
        ("The pull request fixes a critical bug in the authentication module.", "Programming"),
        ("The developer refactored the API to improve response times.", "Programming"),
        ("Python 3.13 introduces several performance improvements.", "Programming"),
        ("The open-source framework reached one million GitHub stars.", "Programming"),
        ("The new type system makes code safer and easier to refactor.", "Programming"),
        # ── Hardware ───────────────────────────────────────────────────────
        ("The new GPU delivers twice the memory bandwidth of its predecessor.", "Hardware"),
        ("The ARM processor achieves remarkable power efficiency.", "Hardware"),
        ("The data center upgraded its server racks with liquid cooling.", "Hardware"),
        ("DDR5 RAM modules are now widely available at competitive prices.", "Hardware"),
        ("The new CPU chipset reduces power consumption by 30 percent.", "Hardware"),
        # ── Nutrition ──────────────────────────────────────────────────────
        ("Increasing protein intake can help maintain muscle mass.", "Nutrition"),
        ("A diet rich in vitamins and minerals supports immune function.", "Nutrition"),
        ("Reducing caloric intake while exercising leads to weight loss.", "Nutrition"),
        ("The new dietary guidelines recommend fewer processed foods.", "Nutrition"),
        ("Eating whole grains and vegetables improves long-term health outcomes.", "Nutrition"),
        # ── Exercise ───────────────────────────────────────────────────────
        ("Regular cardio workouts reduce the risk of heart disease.", "Exercise"),
        ("The gym reported record membership sign-ups in January.", "Exercise"),
        ("A 30-minute morning run significantly boosts energy levels.", "Exercise"),
        ("Yoga and stretching improve flexibility and reduce injury risk.", "Exercise"),
        ("High-intensity interval training burns more calories in less time.", "Exercise"),
        # ── Mental Health ──────────────────────────────────────────────────
        ("Mindfulness meditation has been shown to reduce anxiety levels.", "Mental Health"),
        ("Access to therapy remains a significant challenge for many patients.", "Mental Health"),
        ("Chronic stress can have lasting effects on mental wellbeing.", "Mental Health"),
        ("New apps are making mental health support more accessible.", "Mental Health"),
        ("Cognitive behavioural therapy is an evidence-based treatment for depression.", "Mental Health"),
    ]
    x_train = [text for text, _ in training_data]
    y_train = [label for _, label in training_data]
    return training_data, x_train, y_train


@app.cell
def _(training_data: list[tuple[str, str]], mo):
    max_len = 80
    rows = [
        {"Text": text[:max_len] + ("…" if len(text) > max_len else ""), "Category": label}
        for text, label in training_data
    ]
    mo.accordion(
        {"📋 Show training examples": mo.ui.table(rows, selection=None)},
    )
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
def _(mo):
    mo.md(r"""
    ## Step 2 — Configure and train the pipeline

    Adjust the slider to change the regularisation strength **C** of the
    `LogisticRegression` model.  A higher C means less regularisation (the model
    fits training data more closely); a lower C adds more regularisation.

    The pipeline is **re-trained automatically** whenever you move the slider.
    """)
    return


@app.cell
def _(mo):
    regularisation = mo.ui.slider(
        start=0.01,
        stop=10.0,
        step=0.01,
        value=1.0,
        label="Logistic Regression C (regularisation strength)",
        show_value=True,
    )
    return (regularisation,)


@app.cell
def _(SklearnNodeClassifier, X_train, mo, regularisation, y_train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(
                    C=regularisation.value,
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ],
    )
    pipeline.fit(X_train, y_train)
    node_clf = SklearnNodeClassifier(pipeline)

    mo.callout(
        mo.md(
            f"✅ Pipeline trained with **C = {regularisation.value:.2f}** "
            f"on {len(X_train)} examples across {len(set(y_train))} categories.",
        ),
        kind="success",
    )
    return TfidfVectorizer, node_clf, pipeline


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3 — Classify new text

    Pick a preset or enter your own text.  The `HierarchicalClassifier` will use the
    trained sklearn pipeline to traverse the tree and return the best leaf category.
    """)
    return


@app.cell
def _(mo):
    preset_picker = mo.ui.dropdown(
        options={
            "GPU news": "The latest GPU from NVIDIA sets a new record for compute performance.",
            "Morning run": "A 20-minute jog every morning is one of the best habits you can build.",
            "LLM paper": "The open-source language model was fine-tuned on code and scientific papers.",
            "Diet tips": "Eating a balanced diet with adequate protein supports muscle recovery.",
            "Football transfer": "The club signed the midfielder for a record-breaking transfer fee.",
            "Anxiety support": "Cognitive behavioural therapy is an evidence-based treatment for anxiety.",
            "Python release": "The new Python release ships significant improvements to the type system.",
            "Basketball play-offs": "The team's three-point shooting was unstoppable in the play-offs.",
            "Custom (edit below)": "",
        },
        value="GPU news",
        label="Pick a preset example",
    )
    return (preset_picker,)


@app.cell
def _(mo, preset_picker):
    initial = preset_picker.value or "The latest GPU from NVIDIA sets a new record for compute performance."
    utterance_input = mo.ui.text_area(
        value=initial,
        label="✏️ Text to classify",
        full_width=True,
        rows=3,
    )
    return (utterance_input,)


@app.cell
def _(HierarchicalClassifier, hierarchy, mo, node_clf, utterance_input):
    text = utterance_input.value.strip()
    if text:
        clf = HierarchicalClassifier.from_classifier(node_classifier=node_clf, hierarchy=hierarchy)
        result = clf.classify(text)
        mo.callout(
            mo.md(f"🏷️  **Predicted category:** {result}"),
            kind="success",
        )
    else:
        mo.callout(mo.md("_Enter some text above._"), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4 — Probability breakdown

    The table below shows the raw probability the sklearn model assigns to every
    **leaf category** for the current text, sorted from most to least likely.  This
    lets you see which branches the traversal considered most strongly.
    """)
    return


@app.cell
def _(hierarchy, mo, pipeline, utterance_input):
    text = utterance_input.value.strip()
    if not text:
        mo.callout(mo.md("_Enter text above to see probabilities._"), kind="warn")
    else:
        all_proba = pipeline.predict_proba([text])[0]
        class_to_prob: dict[str, float] = dict(zip(pipeline.classes_, all_proba, strict=True))

        rows = []
        for _top in hierarchy.children:
            for _leaf in _top.children:
                p = class_to_prob.get(_leaf.name, 0.0)
                rows.append(
                    {
                        "Top-level": _top.name,
                        "Leaf category": _leaf.name,
                        "Probability": f"{p:.4f}",
                        "Relative weight": "█" * max(1, round(p * 30)),
                    },
                )
        rows.sort(key=lambda r: float(r["Probability"]), reverse=True)
        mo.ui.table(rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5 — Explore: swap the underlying model

    The `SklearnNodeClassifier` works with **any** sklearn-compatible estimator
    that implements `predict_proba` and exposes `classes_`.  Use the radio buttons
    below to switch between a few classifiers and observe the effect on the results.
    """)
    return


@app.cell
def _(mo):
    model_choice = mo.ui.radio(
        options=["Logistic Regression", "Random Forest", "Naive Bayes"],
        value="Logistic Regression",
        label="Underlying classifier",
    )
    return (model_choice,)


@app.cell
def _(
    SklearnNodeClassifier,
    TfidfVectorizer,
    x_train,
    mo,
    model_choice,
    regularisation,
    y_train,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline as _Pipeline

    if model_choice.value == "Logistic Regression":
        estimator = LogisticRegression(C=regularisation.value, max_iter=500, random_state=42)
    elif model_choice.value == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        estimator = MultinomialNB()

    alt_pipeline = _Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", estimator),
        ],
    )
    alt_pipeline.fit(x_train, y_train)
    alt_node_clf = SklearnNodeClassifier(alt_pipeline)

    mo.callout(
        mo.md(f"✅ **{model_choice.value}** pipeline trained on {len(x_train)} examples."),
        kind="success",
    )
    return (alt_node_clf,)


@app.cell
def _(HierarchicalClassifier, alt_node_clf, hierarchy, mo, utterance_input):
    text = utterance_input.value.strip()
    if text:
        clf = HierarchicalClassifier.from_classifier(node_classifier=alt_node_clf, hierarchy=hierarchy)
        result = clf.classify(text)
        mo.callout(
            mo.md(f"🏷️  **Alternative model prediction:** {result}"),
            kind="info",
        )
    else:
        mo.callout(mo.md("_Enter some text in Step 3 to compare models._"), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next steps

    - **Improve the model**: add more training data, try different vectorisers
      (e.g. `HashingVectorizer`, `CountVectorizer`), or tune hyperparameters with
      `GridSearchCV`.
    - **Zero-shot classification**: `HuggingFaceZeroShotClassifier` from
      `src.integrations.huggingface` requires no labelled training data at all.
    - **LLM-backed classification**: `LLMNodeClassifier` from `src.integrations.llm`
      works with GPT, Claude, Gemini, and 100+ providers via `litellm`.
    - **Save/load the hierarchy**: use `await hierarchy.save("hierarchy.json")` and
      `await HierarchyNode.load("hierarchy.json")`.
    - **Custom classifier from scratch**: see the **custom classifier** notebook for
      a step-by-step guide to implementing your own `NodeClassifier`.
    """)
    return


if __name__ == "__main__":
    app.run()
