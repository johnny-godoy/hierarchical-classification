"""Benchmark path-finding quality and speed across traversal algorithms."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from src.classifier import HierarchicalClassifier
from src.models import HierarchyNode, NodeClassifier
from src.scoring import NegLogScoringStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable


DEFAULT_REPEATS = 12
DEFAULT_OUTPUT = Path(".benchmarks/pathfinding_runs.jsonl")
NUMERIC_EPSILON = 1e-12
ROOT_GREEDY_PROBABILITY = 0.63
ROOT_OPTIMAL_PROBABILITY = 0.37
GREEDY_PATH_PROBABILITY = 0.44
OPTIMAL_PATH_PROBABILITY = 0.97

SerializedProblemResults = dict[str, dict[str, dict[str, float]]]


@dataclasses.dataclass(slots=True)
class BenchmarkProblem:
    """Single benchmark problem definition."""

    name: str
    hierarchy: HierarchyNode[str]
    utterances: list[str]
    probability_tables: dict[str, dict[str, npt.NDArray[np.float32]]]


@dataclasses.dataclass(slots=True)
class AlgorithmResult:
    """Collected benchmark metrics for one algorithm in one problem."""

    optimality_pct: float
    mean_ms: float
    median_ms: float


class LookupNodeClassifier(NodeClassifier[str]):
    """Node classifier backed by deterministic lookup tables."""

    def __init__(self, probability_tables: dict[str, dict[str, npt.NDArray[np.float32]]]) -> None:
        self._probability_tables = probability_tables

    def predict_proba(self, utterance: str, node: HierarchyNode[str]) -> npt.NDArray[np.float32]:
        """Return probabilities for the current utterance and node."""
        try:
            node_table = self._probability_tables[utterance]
            probabilities = node_table[node.name]
        except KeyError as error:
            msg = f"Missing probability table for utterance={utterance!r}, node={node.name!r}."
            raise ValueError(msg) from error
        return probabilities


def build_balanced_hierarchy(depth: int, branching_factor: int, prefix: str) -> HierarchyNode[str]:
    """Build a balanced hierarchy with unique node names."""

    def _build(level: int, path: tuple[int, ...]) -> HierarchyNode[str]:
        if level == depth:
            return HierarchyNode(name=f"{prefix}/leaf/{'.'.join(map(str, path))}")

        children = [_build(level + 1, (*path, index)) for index in range(branching_factor)]
        path_fragment = "root" if not path else ".".join(map(str, path))
        return HierarchyNode(name=f"{prefix}/node/{path_fragment}", children=children)

    return _build(level=0, path=())


def iter_internal_nodes(root: HierarchyNode[str]) -> Iterable[HierarchyNode[str]]:
    """Yield all internal nodes in depth-first order."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            continue
        yield node
        stack.extend(reversed(node.children))


def probability_vector(
    rng: np.random.Generator,
    length: int,
    favored_child: int,
    favored_range: tuple[float, float],
) -> npt.NDArray[np.float32]:
    """Create a normalized probability vector with one favored child."""
    lower_bound, upper_bound = favored_range
    favored_score = rng.uniform(lower_bound, upper_bound)
    noise = rng.uniform(NUMERIC_EPSILON, 0.15, size=length)
    noise[favored_child] = favored_score
    normalized = noise / noise.sum()
    return normalized.astype(np.float32)


def descendant_for_indices(root: HierarchyNode[str], indices: list[int]) -> HierarchyNode[str]:
    """Follow child indices from root and return the reached node."""
    current = root
    for child_index in indices:
        current = current.children[child_index]
    return current


def inject_deceptive_tables(
    hierarchy: HierarchyNode[str],
    utterance_nodes: dict[str, npt.NDArray[np.float32]],
    depth: int,
    branching_factor: int,
) -> None:
    """Inject a local-probability trap where greedy becomes suboptimal."""
    if branching_factor < 2 or depth < 3:
        return
    denominator = branching_factor - 1
    if denominator <= 0:
        return

    greedy_root_index = 0
    optimal_root_index = 1

    root_vector = np.full(branching_factor, NUMERIC_EPSILON, dtype=np.float32)
    root_vector[greedy_root_index] = ROOT_GREEDY_PROBABILITY
    root_vector[optimal_root_index] = ROOT_OPTIMAL_PROBABILITY
    utterance_nodes[hierarchy.name] = root_vector / root_vector.sum()

    greedy_indices: list[int] = [greedy_root_index]
    optimal_indices: list[int] = [optimal_root_index]

    for _ in range(depth - 1):
        greedy_node = descendant_for_indices(hierarchy, greedy_indices)
        optimal_node = descendant_for_indices(hierarchy, optimal_indices)

        greedy_vector = np.full(
            branching_factor,
            (1.0 - GREEDY_PATH_PROBABILITY) / denominator,
            dtype=np.float32,
        )
        greedy_vector[greedy_root_index] = GREEDY_PATH_PROBABILITY
        utterance_nodes[greedy_node.name] = greedy_vector

        optimal_vector = np.full(
            branching_factor,
            (1.0 - OPTIMAL_PATH_PROBABILITY) / denominator,
            dtype=np.float32,
        )
        optimal_vector[optimal_root_index] = OPTIMAL_PATH_PROBABILITY
        utterance_nodes[optimal_node.name] = optimal_vector

        greedy_indices.append(greedy_root_index)
        optimal_indices.append(optimal_root_index)


def create_problem(
    name: str,
    depth: int,
    branching_factor: int,
    utterance_count: int,
    deceptive_fraction: float,
    seed: int,
) -> BenchmarkProblem:
    """Create one benchmark problem with deterministic synthetic probability tables."""
    hierarchy = build_balanced_hierarchy(depth=depth, branching_factor=branching_factor, prefix=name)
    rng = np.random.default_rng(seed)
    utterances = [f"{name}/sample/{index:04d}" for index in range(utterance_count)]
    probability_tables: dict[str, dict[str, npt.NDArray[np.float32]]] = {}

    for utterance in utterances:
        utterance_nodes: dict[str, npt.NDArray[np.float32]] = {}
        for node in iter_internal_nodes(hierarchy):
            favored_child = int(rng.integers(low=0, high=len(node.children)))
            utterance_nodes[node.name] = probability_vector(
                rng=rng,
                length=len(node.children),
                favored_child=favored_child,
                favored_range=(0.50, 0.86),
            )
        probability_tables[utterance] = utterance_nodes

    deceptive_count = int(utterance_count * deceptive_fraction)
    deceptive_utterances = utterances[:deceptive_count]
    for deceptive_utterance in deceptive_utterances:
        inject_deceptive_tables(
            hierarchy=hierarchy,
            utterance_nodes=probability_tables[deceptive_utterance],
            depth=depth,
            branching_factor=branching_factor,
        )

    return BenchmarkProblem(
        name=name,
        hierarchy=hierarchy,
        utterances=utterances,
        probability_tables=probability_tables,
    )


def create_problems() -> list[BenchmarkProblem]:
    """Build realistic benchmark problems of increasing size and complexity."""
    return [
        create_problem(
            name="support-routing-small",
            depth=2,
            branching_factor=3,
            utterance_count=220,
            deceptive_fraction=0.0,
            seed=7,
        ),
        create_problem(
            name="support-routing-medium",
            depth=3,
            branching_factor=3,
            utterance_count=360,
            deceptive_fraction=0.08,
            seed=19,
        ),
        create_problem(
            name="commerce-taxonomy-large",
            depth=4,
            branching_factor=3,
            utterance_count=460,
            deceptive_fraction=0.12,
            seed=31,
        ),
        create_problem(
            name="global-helpdesk-xlarge",
            depth=5,
            branching_factor=3,
            utterance_count=600,
            deceptive_fraction=0.16,
            seed=43,
        ),
    ]


def find_path_to_leaf(root: HierarchyNode[str], leaf_name: str) -> list[HierarchyNode[str]]:
    """Return the node path from root children to the target leaf."""

    def _search(node: HierarchyNode[str], path: list[HierarchyNode[str]]) -> list[HierarchyNode[str]] | None:
        if node.name == leaf_name and node.is_leaf:
            return path

        for child in node.children:
            child_path = _search(child, [*path, child])
            if child_path is not None:
                return child_path

        return None

    result = _search(root, [])
    if result is None:
        msg = f"Leaf {leaf_name!r} does not exist in hierarchy {root.name!r}."
        raise ValueError(msg)
    return result


def path_cost(
    hierarchy: HierarchyNode[str],
    classifier: NodeClassifier[str],
    utterance: str,
    leaf_name: str,
) -> float:
    """Calculate negative-log path cost for a chosen leaf."""
    target_path = find_path_to_leaf(root=hierarchy, leaf_name=leaf_name)
    current_node = hierarchy
    total_cost = 0.0

    for next_node in target_path:
        probabilities = classifier.predict_proba(utterance, current_node)
        child_index = current_node.children.index(next_node)
        probability = float(probabilities[child_index])
        if probability <= 0:
            return math.inf
        total_cost -= math.log(probability)
        current_node = next_node

    return total_cost


def greedy_classify(hierarchy: HierarchyNode[str], classifier: NodeClassifier[str], utterance: str) -> str:
    """Classify by locally maximizing probability at each step."""
    current = hierarchy
    while not current.is_leaf:
        probabilities = classifier.predict_proba(utterance, current)
        best_index = int(np.argmax(probabilities))
        if float(probabilities[best_index]) <= 0:
            msg = "No leaf node found in the hierarchy."
            raise ValueError(msg)
        current = current.children[best_index]
    return current.name


def brute_force_classify(hierarchy: HierarchyNode[str], classifier: NodeClassifier[str], utterance: str) -> str:
    """Classify by exhaustive path search over all leaf routes."""
    best_leaf: str | None = None
    best_cost = math.inf

    def _search(node: HierarchyNode[str], current_cost: float) -> None:
        nonlocal best_leaf, best_cost

        if node.is_leaf:
            if current_cost < best_cost:
                best_cost = current_cost
                best_leaf = node.name
            return

        probabilities = classifier.predict_proba(utterance, node)
        for child_index, child in enumerate(node.children):
            probability = float(probabilities[child_index])
            if probability <= 0:
                continue
            _search(child, current_cost - math.log(probability))

    _search(hierarchy, 0.0)
    if best_leaf is None:
        msg = "No leaf node found in the hierarchy."
        raise ValueError(msg)

    return best_leaf


def build_main_classifier(hierarchy: HierarchyNode[str], classifier: NodeClassifier[str]) -> HierarchicalClassifier:
    """Build a reusable instance for the repository's main best-first algorithm."""
    scoring = NegLogScoringStrategy(classifier)
    return HierarchicalClassifier(scoring_strategy=scoring, hierarchy=hierarchy)


def run_timed_predictions(
    algorithm_name: str,
    hierarchy: HierarchyNode[str],
    classifier: NodeClassifier[str],
    utterances: list[str],
    repeats: int,
) -> list[float]:
    """Measure prediction times in milliseconds."""
    durations_ms: list[float] = []
    main_classifier: HierarchicalClassifier | None = None
    if algorithm_name == "main":
        main_classifier = build_main_classifier(hierarchy=hierarchy, classifier=classifier)
    for _ in range(repeats):
        for utterance in utterances:
            start = perf_counter()
            if algorithm_name == "main":
                if main_classifier is None:
                    msg = "Main classifier was not initialized."
                    raise ValueError(msg)
                main_classifier.classify(utterance)
            elif algorithm_name == "greedy":
                greedy_classify(hierarchy=hierarchy, classifier=classifier, utterance=utterance)
            elif algorithm_name == "bruteforce":
                brute_force_classify(hierarchy=hierarchy, classifier=classifier, utterance=utterance)
            else:
                msg = f"Unknown algorithm {algorithm_name!r}."
                raise ValueError(msg)
            end = perf_counter()
            durations_ms.append((end - start) * 1000.0)

    return durations_ms


def compute_optimality(
    hierarchy: HierarchyNode[str],
    classifier: NodeClassifier[str],
    utterances: list[str],
    algorithm_name: str,
) -> float:
    """Compute percent of utterances whose score matches brute-force optimal."""
    optimal_matches = 0
    main_classifier: HierarchicalClassifier | None = None
    if algorithm_name == "main":
        main_classifier = build_main_classifier(hierarchy=hierarchy, classifier=classifier)

    for utterance in utterances:
        brute_force_leaf = brute_force_classify(hierarchy=hierarchy, classifier=classifier, utterance=utterance)
        brute_force_cost = path_cost(
            hierarchy=hierarchy,
            classifier=classifier,
            utterance=utterance,
            leaf_name=brute_force_leaf,
        )

        if algorithm_name == "main":
            if main_classifier is None:
                msg = "Main classifier was not initialized."
                raise ValueError(msg)
            predicted_leaf = main_classifier.classify(utterance)
        elif algorithm_name == "greedy":
            predicted_leaf = greedy_classify(hierarchy=hierarchy, classifier=classifier, utterance=utterance)
        elif algorithm_name == "bruteforce":
            predicted_leaf = brute_force_leaf
        else:
            msg = f"Unknown algorithm {algorithm_name!r}."
            raise ValueError(msg)

        predicted_cost = path_cost(
            hierarchy=hierarchy,
            classifier=classifier,
            utterance=utterance,
            leaf_name=predicted_leaf,
        )
        if math.isclose(predicted_cost, brute_force_cost, rel_tol=0.0, abs_tol=1e-9):
            optimal_matches += 1

    return (optimal_matches / len(utterances)) * 100.0


def evaluate_algorithm(
    algorithm_name: str,
    hierarchy: HierarchyNode[str],
    classifier: NodeClassifier[str],
    utterances: list[str],
    repeats: int,
) -> AlgorithmResult:
    """Evaluate one algorithm and return optimality and runtime metrics."""
    optimality = compute_optimality(
        hierarchy=hierarchy,
        classifier=classifier,
        utterances=utterances,
        algorithm_name=algorithm_name,
    )
    durations_ms = run_timed_predictions(
        algorithm_name=algorithm_name,
        hierarchy=hierarchy,
        classifier=classifier,
        utterances=utterances,
        repeats=repeats,
    )

    return AlgorithmResult(
        optimality_pct=optimality,
        mean_ms=statistics.fmean(durations_ms),
        median_ms=statistics.median(durations_ms),
    )


def resolve_git_commit(repo_root: Path) -> str:
    """Resolve current git commit hash without shelling out."""
    git_dir = repo_root / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return "unknown"

    head_text = head_file.read_text(encoding="utf-8").strip()
    ref_prefix = "ref: "
    if not head_text.startswith(ref_prefix):
        return head_text

    ref_path = head_text.removeprefix(ref_prefix)
    commit_file = git_dir / ref_path
    if commit_file.exists():
        return commit_file.read_text(encoding="utf-8").strip()

    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return "unknown"

    for line in packed_refs.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        line_parts = line.split(" ", maxsplit=1)
        if len(line_parts) != 2:
            continue
        sha, reference = line_parts
        if reference == ref_path:
            return sha
    return "unknown"


def run_suite(repeats: int) -> dict[str, dict[str, AlgorithmResult]]:
    """Run the benchmark suite and return per-problem metrics."""
    problems = create_problems()
    suite_results: dict[str, dict[str, AlgorithmResult]] = {}

    for problem in problems:
        classifier = LookupNodeClassifier(problem.probability_tables)
        problem_results = {
            "main": evaluate_algorithm(
                algorithm_name="main",
                hierarchy=problem.hierarchy,
                classifier=classifier,
                utterances=problem.utterances,
                repeats=repeats,
            ),
            "greedy": evaluate_algorithm(
                algorithm_name="greedy",
                hierarchy=problem.hierarchy,
                classifier=classifier,
                utterances=problem.utterances,
                repeats=repeats,
            ),
            "bruteforce": evaluate_algorithm(
                algorithm_name="bruteforce",
                hierarchy=problem.hierarchy,
                classifier=classifier,
                utterances=problem.utterances,
                repeats=repeats,
            ),
        }
        suite_results[problem.name] = problem_results

    return suite_results


def append_run(
    run_output_path: Path,
    run_payload: dict[str, str | int | SerializedProblemResults],
) -> None:
    """Append run payload to a JSONL file."""
    run_output_path.parent.mkdir(parents=True, exist_ok=True)
    with run_output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_payload))
        handle.write("\n")


def serialize_results(
    suite_results: dict[str, dict[str, AlgorithmResult]],
) -> SerializedProblemResults:
    """Convert dataclass suite results into plain JSON serializable dictionaries."""
    serialized: dict[str, dict[str, dict[str, float]]] = {}
    for problem_name, algorithms in suite_results.items():
        serialized[problem_name] = {
            algorithm_name: {
                "optimality_pct": metrics.optimality_pct,
                "mean_ms": metrics.mean_ms,
                "median_ms": metrics.median_ms,
            }
            for algorithm_name, metrics in algorithms.items()
        }
    return serialized


def print_summary(suite_results: dict[str, dict[str, AlgorithmResult]]) -> None:
    """Print benchmark results in a compact table-like format."""
    header = (
        "problem",
        "algorithm",
        "optimality_pct",
        "mean_ms",
        "median_ms",
    )
    lines = ["\t".join(header)]

    for problem_name, algorithms in suite_results.items():
        for algorithm_name, metrics in algorithms.items():
            row = (
                problem_name,
                algorithm_name,
                f"{metrics.optimality_pct:.2f}",
                f"{metrics.mean_ms:.4f}",
                f"{metrics.median_ms:.4f}",
            )
            lines.append("\t".join(row))

    sys.stdout.write("\n".join(lines))
    sys.stdout.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="How many timing rounds to execute for each utterance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSONL file where benchmark runs are appended.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute benchmark suite and persist run results."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    suite_results = run_suite(repeats=args.repeats)

    run_payload = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "commit": resolve_git_commit(repo_root),
        "repeats": args.repeats,
        "problems": serialize_results(suite_results),
    }

    append_run(run_output_path=args.output, run_payload=run_payload)
    print_summary(suite_results)
    sys.stdout.write(f"\nSaved run to {args.output}\n")


if __name__ == "__main__":
    main()
