# hierarchical-classification

Lightweight hierarchical classification for Python. The library lets you define a category tree, attach a node-level classifier, and route each input to the best matching leaf label.

It is designed for cases where a flat label space is too coarse or too large, for example:

- support ticket triage
- document taxonomy assignment
- intent routing across nested categories
- topic classification with broad and fine-grained labels

The project currently provides:

- a tree model built around `HierarchyNode`
- a best-first hierarchical traversal in `HierarchicalClassifier`
- a classifier protocol for custom backends
- ready-to-use integrations for scikit-learn, Hugging Face zero-shot models, and LiteLLM
- marimo notebooks that show both a custom classifier and an sklearn-based workflow

## Installation

Python 3.12+ is required.

### Install from a package index with `uv add`

If the package is published to the index you use, install the core package like this:

```bash
uv add hierarchical-classification
```

Install an integration extra the same way:

```bash
uv add "hierarchical-classification[sklearn]"
uv add "hierarchical-classification[huggingface]"
uv add "hierarchical-classification[llm]"
```

If you want notebook tooling as well:

```bash
uv add "hierarchical-classification[sklearn,notebooks]"
```

If you want the same extras flow directly from Git before publishing, use a direct reference:

```bash
uv add "hierarchical-classification[sklearn] @ git+https://github.com/johnny-godoy/hierarchical-classification"
```

### Install from source with `uv`

```bash
git clone https://github.com/johnny-godoy/hierarchical-classification
cd hierarchical-classification
uv sync
```

Install source checkouts with extras when you need a specific backend:

```bash
uv sync --extra sklearn
uv sync --extra huggingface
uv sync --extra llm
uv sync --extra notebooks
```

### Optional integration dependencies

The project declares optional dependencies in `pyproject.toml`, so users do not need to install backend packages by hand if they install the right extra.

The extras are:

```bash
hierarchical-classification[sklearn]
hierarchical-classification[huggingface]
hierarchical-classification[llm]
hierarchical-classification[notebooks]
```

They map to these backends:

- `scikit-learn` for classical ML pipelines
- `transformers` and a compatible runtime such as `torch` for zero-shot classification
- `litellm` for hosted or local LLM-based routing
- `marimo` to run the notebooks in `notebooks/`

## LLM and Agent Friendly Usage

If you're using  coding agents to interact with this library, keep a compact machine-readable brief in the repository root:

- `llms.txt`: short, high-signal usage contract
- `llms-full.txt`: longer version with caveats and examples

Both files are included in this repository and summarize exactly what agents usually need:

- install command (`uv add` or `uv sync --extra ...`)
- core imports (`HierarchyNode`, `HierarchicalClassifier`, backend adapters)
- required method contract (`predict_proba(utterance, node)`)
- backend-specific constraints (for example, sklearn `classes_` coverage)

Recommended prompt pattern for tools/agents:

```text
Use this repository's llms.txt as the source of truth.
Prefer the smallest working hierarchy first.
Use HierarchicalClassifier.from_classifier(...).
Only pull optional extras for the backend you selected.
```

## How it works

At each non-leaf node, the classifier scores that node's children. The traversal keeps exploring the lowest-cost path until it finds the best leaf.

```text
input text
   |
   v
root node
   |
score root children
   |
pick the most promising branch
   |
score branch children
   |
repeat until a leaf is reached
   |
   v
predicted leaf label
```

This structure is useful when:

- coarse categories should be decided before fine-grained ones
- different label groups are easier to distinguish locally than globally
- you want a reusable wrapper around different scoring backends

## Quick Start

Start with a tiny hierarchy and a custom classifier, then swap in one of the built-in integrations when you want a stronger backend.

### 1. Define a hierarchy

```python
from src.models import HierarchyNode

hierarchy = HierarchyNode(
	name="root",
	children=[
		HierarchyNode(
			name="animal",
			children=[
				HierarchyNode(name="cat"),
				HierarchyNode(name="dog"),
			],
		),
		HierarchyNode(
			name="vehicle",
			children=[
				HierarchyNode(name="car"),
				HierarchyNode(name="bike"),
			],
		),
	],
)
```

### 2. Implement a custom node classifier

`NodeClassifier` only needs a `predict_proba(utterance, node)` method that returns one probability per child of the current node.

```python
import numpy as np

from src.classifier import HierarchicalClassifier
from src.models import HierarchyNode, NodeClassifier


class KeywordNodeClassifier(NodeClassifier):
	def predict_proba(self, utterance: str, node: HierarchyNode) -> np.ndarray:
		text = utterance.lower()
		child_names = [child.name for child in node.children]

		if child_names == ["animal", "vehicle"]:
			scores = np.array([
				0.9 if any(word in text for word in ["cat", "dog", "pet"]) else 0.1,
				0.9 if any(word in text for word in ["car", "bike", "road"]) else 0.1,
			], dtype=np.float32)
		elif child_names == ["cat", "dog"]:
			scores = np.array([
				0.9 if "cat" in text else 0.1,
				0.9 if "dog" in text else 0.1,
			], dtype=np.float32)
		elif child_names == ["car", "bike"]:
			scores = np.array([
				0.9 if "car" in text else 0.1,
				0.9 if "bike" in text else 0.1,
			], dtype=np.float32)
		else:
			scores = np.ones(len(node.children), dtype=np.float32)

		return scores / scores.sum()


classifier = HierarchicalClassifier.from_classifier(
	KeywordNodeClassifier(),
	hierarchy,
)

print(classifier.classify("the dog barked loudly"))
# dog
```

## Usage Examples

### scikit-learn integration

Use `SklearnNodeClassifier` to adapt any fitted sklearn classifier or pipeline that exposes `predict_proba` and `classes_`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.classifier import HierarchicalClassifier
from src.integrations.sklearn import SklearnNodeClassifier

X_train = [
	"cats purr and sleep a lot",
	"dogs bark and wag their tails",
	"cars drive on highways",
	"bikes are popular in cities",
	"animals need food and water",
	"vehicles can be used for transport",
]

y_train = ["cat", "dog", "car", "bike", "animal", "vehicle"]

pipeline = Pipeline(
	[
		("tfidf", TfidfVectorizer()),
		("clf", LogisticRegression(max_iter=1000)),
	]
)
pipeline.fit(X_train, y_train)

node_classifier = SklearnNodeClassifier(pipeline)
classifier = HierarchicalClassifier.from_classifier(node_classifier, hierarchy)

print(classifier.classify("my bike has a flat tire"))
# bike
```

Important: the fitted classifier's `classes_` must include every child label that can appear anywhere in the hierarchy.

### Hugging Face zero-shot integration

Use `HuggingFaceZeroShotClassifier` when you want label routing without training a local classifier.

```python
from src.classifier import HierarchicalClassifier
from src.integrations.huggingface import HuggingFaceZeroShotClassifier

node_classifier = HuggingFaceZeroShotClassifier(
	"facebook/bart-large-mnli"
)
classifier = HierarchicalClassifier.from_classifier(node_classifier, hierarchy)

print(classifier.classify("I adopted a playful kitten"))
```

This integration uses the child node names directly as candidate labels.

### LiteLLM integration

Use `LLMNodeClassifier` when you want an LLM to choose the best child label at each node.

```python
from src.classifier import HierarchicalClassifier
from src.integrations.llm import LLMNodeClassifier

node_classifier = LLMNodeClassifier("gpt-4o-mini")
classifier = HierarchicalClassifier.from_classifier(node_classifier, hierarchy)

print(classifier.classify("I need to replace the chain on my bike"))
```

Set the provider-specific credentials before use, for example:

```bash
set OPENAI_API_KEY=your-key
```

You can also point LiteLLM at local providers such as Ollama by changing the model string and any required connection parameters.

## Notebooks

The repository includes interactive marimo notebooks in `notebooks/`:

- `notebooks/custom_classifier.py`: builds a custom keyword-based `NodeClassifier`
- `notebooks/sklearn_integration.py`: trains and evaluates an sklearn pipeline inside the hierarchical wrapper

### Run a notebook

Install marimo if you have not already:

```bash
uv sync --extra notebooks --extra sklearn
```

Then launch a notebook:

```bash
uv run marimo edit notebooks/custom_classifier.py
```

or:

```bash
uv run marimo edit notebooks/sklearn_integration.py
```

Both notebooks add the project root to `sys.path` automatically, so they can be run from the repository root without extra path setup.

## Project Layout

```text
src/
  classifier.py            Hierarchical traversal logic
  models.py                Core tree and classifier protocols
  integrations/
	sklearn.py             scikit-learn adapter
	huggingface.py         zero-shot transformers adapter
	llm.py                 LiteLLM adapter
notebooks/
  custom_classifier.py     marimo notebook for a custom classifier
  sklearn_integration.py   marimo notebook for sklearn usage
tests/
  unit/
  integration/
```

## Development

Install the project, then run the test suite:

```bash
uv run pytest
```

To run a narrower slice:

```bash
uv run pytest tests/integration/test_full_pipeline.py
```

## Current Status

This repository already includes tested integration coverage for:

- custom classifiers via the `NodeClassifier` protocol
- sklearn-based probabilistic classifiers
- Hugging Face zero-shot pipelines
- LiteLLM-backed model selection

The public package metadata is still minimal, so this README documents the current codebase rather than a published PyPI release flow.

## License

See `LICENSE.md`.
