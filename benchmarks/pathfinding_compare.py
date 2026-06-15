"""Compare path-finding benchmark runs and surface the fastest fully-optimal run."""

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = Path(".benchmarks/pathfinding_runs.jsonl")
DEFAULT_PLOT_DIR = Path(".benchmarks/reports")
OPTIMALITY_TARGET = 100.0
MIN_RUNS_FOR_DELTA = 2

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


@dataclass(slots=True)
class RunSummary:
    """Parsed benchmark run payload with computed convenience fields."""

    timestamp_utc: str
    timestamp: dt.datetime
    commit: str
    repeats: int
    problems: dict[str, dict[str, dict[str, float]]]

    @property
    def short_commit(self) -> str:
        """Return a short commit hash used in compact labels."""
        return self.commit[:7] if self.commit != "unknown" else "unknown"

    @property
    def label(self) -> str:
        """Return compact identifier for logs and plots."""
        stamp = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"{stamp} | {self.short_commit}"

    @property
    def main_mean_ms(self) -> float:
        """Return average main mean_ms across all benchmark problems."""
        values = [problem_metrics["main"]["mean_ms"] for problem_metrics in self.problems.values()]
        return statistics.fmean(values)

    @property
    def main_median_ms(self) -> float:
        """Return average main median_ms across all benchmark problems."""
        values = [problem_metrics["main"]["median_ms"] for problem_metrics in self.problems.values()]
        return statistics.fmean(values)

    @property
    def main_optimality_min_pct(self) -> float:
        """Return the minimum main optimality across benchmark problems."""
        values = [problem_metrics["main"]["optimality_pct"] for problem_metrics in self.problems.values()]
        return min(values)


@dataclass(slots=True)
class ParsedRuns:
    """Container for loaded and sorted runs."""

    runs: list[RunSummary]

    @property
    def qualified(self) -> list[RunSummary]:
        """Return runs where main remains fully optimal on all problems."""
        return [run for run in self.runs if is_main_fully_optimal(run)]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for run comparison.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSONL file produced by benchmarks/pathfinding_benchmark.py.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top qualified runs to print in the ranking.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory where interactive HTML plots are written.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate interactive HTML plots (default: true).",
    )
    return parser.parse_args()


def load_runs(path: Path) -> ParsedRuns:
    """Load benchmark runs from JSONL and sort by timestamp.

    Parameters
    ----------
    path : Path
        JSONL file produced by the benchmark runner.

    Returns
    -------
    ParsedRuns
        Loaded run payloads sorted by timestamp.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If a JSONL line has an invalid payload schema.
    """
    if not path.exists():
        msg = f"Input file does not exist: {path}"
        raise FileNotFoundError(msg)

    loaded: list[RunSummary] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                timestamp_utc = str(payload["timestamp_utc"])
                timestamp = dt.datetime.fromisoformat(timestamp_utc)
                run = RunSummary(
                    timestamp_utc=timestamp_utc,
                    timestamp=timestamp,
                    commit=str(payload["commit"]),
                    repeats=int(payload["repeats"]),
                    problems=dict(payload["problems"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                msg = f"Invalid run payload at line {line_number} in {path}: {error}"
                raise ValueError(msg) from error
            loaded.append(run)

    loaded.sort(key=lambda run: run.timestamp)
    return ParsedRuns(runs=loaded)


def is_main_fully_optimal(run: RunSummary) -> bool:
    """Return true when main is effectively 100% optimal on all problems.

    Parameters
    ----------
    run : RunSummary
        Run to validate.

    Returns
    -------
    bool
        ``True`` when every problem reports 100% main optimality.
    """
    return all(
        math.isclose(problem_metrics["main"]["optimality_pct"], OPTIMALITY_TARGET, rel_tol=0.0, abs_tol=1e-9)
        for problem_metrics in run.problems.values()
    )


def print_summary(parsed: ParsedRuns, top: int) -> None:
    """Print a compact report showing best fully-optimal runs."""
    all_runs = parsed.runs
    qualified_runs = sorted(parsed.qualified, key=lambda run: run.main_mean_ms)

    sys.stdout.write(f"Loaded runs: {len(all_runs)}\n")
    sys.stdout.write(f"Qualified (main optimality=100% on all problems): {len(qualified_runs)}\n\n")

    if not all_runs:
        sys.stdout.write("No runs found.\n")
        return

    if qualified_runs:
        winner = qualified_runs[0]
        sys.stdout.write("Best qualified run:\n")
        sys.stdout.write(
            "  "
            f"{winner.label} | "
            f"main_mean_ms={winner.main_mean_ms:.6f} | "
            f"main_median_ms={winner.main_median_ms:.6f} | "
            f"main_min_optimality={winner.main_optimality_min_pct:.2f}%\n\n",
        )

        sys.stdout.write("Top qualified runs (lower main_mean_ms is better):\n")
        sys.stdout.write("rank | timestamp_utc | commit | main_mean_ms | main_median_ms\n")
        for index, run in enumerate(qualified_runs[: max(top, 1)], start=1):
            sys.stdout.write(
                f"{index} | {run.timestamp_utc} | {run.short_commit} | "
                f"{run.main_mean_ms:.6f} | {run.main_median_ms:.6f}\n",
            )
        sys.stdout.write("\n")
    else:
        sys.stdout.write("No qualified runs match the 'main must stay at 100% optimality' rule.\n\n")

    if len(all_runs) >= MIN_RUNS_FOR_DELTA:
        previous, latest = all_runs[-2], all_runs[-1]
        delta_ms = latest.main_mean_ms - previous.main_mean_ms
        direction = "faster" if delta_ms < 0 else "slower" if delta_ms > 0 else "equal"

        sys.stdout.write("Latest vs previous (main mean over problems):\n")
        sys.stdout.write(f"  previous: {previous.label} | {previous.main_mean_ms:.6f} ms\n")
        sys.stdout.write(f"  latest  : {latest.label} | {latest.main_mean_ms:.6f} ms\n")
        sys.stdout.write(f"  delta   : {delta_ms:+.6f} ms ({direction})\n")


def generate_plots(parsed: ParsedRuns, plot_dir: Path) -> list[Path]:
    """Generate interactive HTML plots and return written file paths.

    Parameters
    ----------
    parsed : ParsedRuns
        Parsed run payloads.
    plot_dir : Path
        Output directory for HTML reports.

    Returns
    -------
    list[Path]
        Paths of written HTML files. Empty when plotting is unavailable.
    """
    if go is None:
        sys.stdout.write("\nSkipping plots: plotly is not installed.\n")
        return []

    runs = parsed.runs
    if not runs:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    labels = [run.label for run in runs]
    means = [run.main_mean_ms for run in runs]
    qualified_flags = [is_main_fully_optimal(run) for run in runs]
    colors = ["#1f77b4" if is_qualified else "#d62728" for is_qualified in qualified_flags]

    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=labels,
            y=means,
            mode="lines+markers",
            marker={"size": 10, "color": colors},
            line={"width": 2, "color": "#6c757d"},
            text=["qualified" if flag else "not qualified" for flag in qualified_flags],
            hovertemplate=("run=%{x}<br>main_mean_ms=%{y:.6f}<br>status=%{text}<extra></extra>"),
            name="main_mean_ms",
        ),
    )
    trend.update_layout(
        title="Main Algorithm Speed Across Runs",
        xaxis_title="run",
        yaxis_title="avg main mean_ms across problems",
    )

    trend_path = plot_dir / "pathfinding_main_speed_trend.html"
    trend.write_html(trend_path, include_plotlyjs="cdn")
    written.append(trend_path)

    problem_names = list(runs[0].problems)
    per_problem = go.Figure()
    for run in runs:
        per_problem.add_trace(
            go.Bar(
                name=run.label,
                x=problem_names,
                y=[run.problems[problem]["main"]["mean_ms"] for problem in problem_names],
                hovertemplate=("problem=%{x}<br>main_mean_ms=%{y:.6f}<br>run=" + run.label + "<extra></extra>"),
            ),
        )

    per_problem.update_layout(
        barmode="group",
        title="Main Mean Runtime by Problem and Run",
        xaxis_title="problem",
        yaxis_title="main mean_ms",
    )

    problem_path = plot_dir / "pathfinding_main_problem_breakdown.html"
    per_problem.write_html(problem_path, include_plotlyjs="cdn")
    written.append(problem_path)

    return written


def main() -> None:
    """Execute benchmark run comparison and optionally write plots."""
    args = parse_args()
    parsed = load_runs(args.input)
    print_summary(parsed, top=args.top)

    if args.plots:
        written_plots = generate_plots(parsed, plot_dir=args.plot_dir)
        if written_plots:
            sys.stdout.write("\nWrote interactive plots:\n")
            for path in written_plots:
                sys.stdout.write(f"  - {path}\n")


if __name__ == "__main__":
    main()
