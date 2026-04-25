from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from clip_image_similarity.packed_distances import PackedDistances
from clip_image_similarity.utils import log, plural
from metrics.precision_recall import (
    DatasetPrecisionRecallResult,
    dataset_precision_recall_at_k,
    select_k_values,
    select_k_values_log,
)
from metrics.series_indices import load_series_indices, validate_series_indices


def _sanitize_filename_stem(name: str) -> str:
    """Convert an arbitrary series name into a safe filename stem."""
    name = name.strip()
    if not name:
        return "series"
    name = re.sub(r"[\\/]+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "series"


def _resolve_inputs(pairwise_output_dir: Path) -> Tuple[Path, Path]:
    """Resolve required input paths from a CLI run output directory.

    Args:
        pairwise_output_dir: Output directory produced by `clip_image_similarity.cli`.
    Returns:
        Tuple (distances_npz_path, series_to_indices_json_path).
    Raises:
        FileNotFoundError: If expected files are missing.
    """
    dist_path = pairwise_output_dir / "evaluation_results" / "pairwise_distances.npz"
    series_path = pairwise_output_dir / "series_to_indices.json"
    if not dist_path.is_file():
        raise FileNotFoundError(f"Distances file not found: {dist_path}")
    if not series_path.is_file():
        raise FileNotFoundError(f"Series indices file not found: {series_path}")
    return dist_path, series_path


def _make_run_output_dir(pairwise_output_dir: Path) -> Path:
    """Create and return the output directory for a precision/recall plot run."""
    graphs_root = pairwise_output_dir / "graphs"
    graphs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = graphs_root / f"precision_recall_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _labeled_union(series_to_indices: Dict[str, List[int]]) -> List[int]:
    """Compute the union of all labeled indices across all series."""
    labeled: Set[int] = set()
    for idxs in series_to_indices.values():
        labeled.update(int(x) for x in idxs)
    return sorted(labeled)


def build_predictions_for_queries(
    dist: PackedDistances, queries: Sequence[int], *, show_progress: bool = False
) -> Dict[int, List[int]]:
    """Build ranked neighbor lists by sorting cosine distances for each query.

    Args:
        dist: PackedDistances accessor for flattened upper-tri distances.
        queries: Query indices to build predictions for.
        show_progress: If True, show a progress bar via tqdm.
    Returns:
        Dict mapping query index -> list of predicted neighbor indices sorted by distance ascending.
    """
    preds_by_query: Dict[int, List[int]] = {}
    n = dist.n
    query_iter = queries
    if show_progress:
        from tqdm import tqdm  # type: ignore

        query_iter = tqdm(list(queries), desc="Building ranked predictions", unit="query")
    for q in query_iter:
        pairs: List[Tuple[int, float]] = []
        for j in range(n):
            if j == q:
                continue
            pairs.append((j, dist.distance(q, j)))
        pairs.sort(key=lambda kv: kv[1])
        preds_by_query[q] = [idx for idx, _ in pairs]
    return preds_by_query


def _plot_precision_recall_curve(
    *,
    output_path: Path,
    title: str,
    ks: List[int],
    precision_values: List[float],
    recall_values: List[float],
) -> None:
    """Plot precision and recall vs k on the same axes and save to a PNG.

    Args:
        output_path: Destination PNG path.
        title: Plot title.
        ks: List of k values.
        precision_values: Precision@k values aligned with ks.
        recall_values: Recall@k values aligned with ks.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: WPS433

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, precision_values, color="tab:blue", label="precision")
    ax.plot(ks, recall_values, color="tab:orange", label="recall")
    ax.set_title(title)
    ax.set_xlabel("k (number of nearest neighbors considered)")
    ax.set_ylabel("value")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_precision_vs_recall_curve(
    *,
    output_path: Path,
    title: str,
    precision_values: List[float],
    recall_values: List[float],
) -> None:
    """Plot precision (y) vs recall (x) and save to a PNG.

    Points correspond to different k values (in increasing k order).

    Args:
        output_path: Destination PNG path.
        title: Plot title.
        precision_values: Precision@k values.
        recall_values: Recall@k values.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: WPS433

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall_values, precision_values, marker="o", color="tab:blue", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@dataclass(frozen=True)
class PrecisionRecallPlotPaths:
    """Where precision/recall plots are written for a run."""

    run_dir: Path
    dataset_plot: Path
    dataset_precision_vs_recall_plot: Path
    per_series_dir: Path


def _build_output_paths(run_dir: Path) -> PrecisionRecallPlotPaths:
    """Construct output paths for a run under the given directory."""
    per_series_dir = run_dir / "series"
    per_series_dir.mkdir(parents=True, exist_ok=True)
    return PrecisionRecallPlotPaths(
        run_dir=run_dir,
        dataset_plot=run_dir / "all_series.png",
        dataset_precision_vs_recall_plot=run_dir / "all_series_precision_vs_recall.png",
        per_series_dir=per_series_dir,
    )


class PrecisionRecallPlotRunner:
    """End-to-end runner that builds predictions and plots PR@k curves."""

    def __init__(
        self,
        *,
        max_predictions: int | None = None,
        steps: int = 1,
        num_k: int | None = None,
    ):
        """Create a PrecisionRecallPlotRunner.

        Args:
            max_predictions: Maximum k to compute up to (inclusive). If None, uses N-1.
            steps: Step size for selecting k values between [1, max_predictions].
            num_k: If provided, selects k values in log space using approximately num_k samples.
        """
        self._max_predictions = int(max_predictions) if max_predictions is not None else None
        self._steps = int(steps)
        self._num_k = int(num_k) if num_k is not None else None

    def run(self, *, pairwise_output_dir: Path) -> PrecisionRecallPlotPaths:
        """Generate PR@k curves and save plots under pairwise_output_dir/graphs/.

        Args:
            pairwise_output_dir: Output directory produced by `clip_image_similarity.cli`.
        Returns:
            PrecisionRecallPlotPaths describing where plots were written.
        """
        pairwise_output_dir = pairwise_output_dir.resolve()
        dist_path, series_path = _resolve_inputs(pairwise_output_dir)
        run_dir = _make_run_output_dir(pairwise_output_dir)
        paths = _build_output_paths(run_dir)

        log(f"Loading distances from {dist_path}...")
        dist = PackedDistances.load(dist_path)
        log(f"Loaded packed distances for {plural(dist.n, 'image')} (dtype={dist.dtype}).")

        log(f"Loading series indices from {series_path}...")
        series_to_indices = load_series_indices(series_path)
        series_to_indices = validate_series_indices(series_to_indices, dist.n)
        labeled_union = _labeled_union(series_to_indices)
        log(f"Found {plural(len(series_to_indices), 'series')} and {plural(len(labeled_union), 'labeled image')}.")

        max_possible = dist.n - 1
        max_predictions = self._max_predictions if self._max_predictions is not None else max_possible
        if max_predictions > max_possible:
            raise ValueError(
                f"max_predictions ({max_predictions}) exceeds maximum possible (N-1={max_possible})."
            )
        if self._num_k is not None and self._steps != 1:
            raise ValueError("Provide either steps or num_k, not both.")
        ks = (
            select_k_values_log(max_predictions=max_predictions, num_k=self._num_k)
            if self._num_k is not None
            else select_k_values(max_predictions=max_predictions, steps=self._steps)
        )

        log("Building ranked predictions per query by sorting pairwise distances...")
        preds_by_query = build_predictions_for_queries(dist, labeled_union, show_progress=True)

        log("Computing dataset-level and per-series Precision@k / Recall@k...")
        result: DatasetPrecisionRecallResult = dataset_precision_recall_at_k(
            preds_by_query=preds_by_query,
            series_to_indices=series_to_indices,
            max_predictions=max_predictions,
            ks=ks,
            show_progress=True,
        )

        ks = [p.k for p in result.dataset_curve]
        dataset_precision = [p.precision.value for p in result.dataset_curve]
        dataset_recall = [p.recall.value for p in result.dataset_curve]
        _plot_precision_recall_curve(
            output_path=paths.dataset_plot,
            title="All Series (precision and recall vs k)",
            ks=ks,
            precision_values=dataset_precision,
            recall_values=dataset_recall,
        )
        log(f"Wrote dataset curve: {paths.dataset_plot}")
        _plot_precision_vs_recall_curve(
            output_path=paths.dataset_precision_vs_recall_plot,
            title="All Series (precision vs recall)",
            precision_values=dataset_precision,
            recall_values=dataset_recall,
        )
        log(f"Wrote dataset PR curve: {paths.dataset_precision_vs_recall_plot}")

        for series_name, curve in result.series_curves.items():
            ks_s = [p.k for p in curve]
            prec_s = [p.precision.value for p in curve]
            rec_s = [p.recall.value for p in curve]
            stem = _sanitize_filename_stem(series_name)
            out_path = paths.per_series_dir / f"{stem}.png"
            _plot_precision_recall_curve(
                output_path=out_path,
                title=f"{series_name} (precision and recall vs k)",
                ks=ks_s,
                precision_values=prec_s,
                recall_values=rec_s,
            )
            log(f"Wrote series curve: {out_path}")

        log(f"Done. Outputs saved under {paths.run_dir}")
        return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for precision/recall plotting."""
    parser = argparse.ArgumentParser(
        description="Plot Precision@k and Recall@k curves from pairwise distances and series labels."
    )
    parser.add_argument(
        "--pairwise-output-dir",
        required=True,
        help="Output directory produced by clip_image_similarity.cli (contains evaluation_results/pairwise_distances.npz and series_to_indices.json).",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=None,
        help="Maximum k to compute up to (inclusive). If omitted, uses N-1.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Step size for selecting k values between [1, max_predictions] (default: 1).",
    )
    group.add_argument(
        "--num-k",
        type=int,
        default=None,
        help="Number of k values to evaluate (log-spaced between 1 and max_predictions).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for `python -m metrics.precision_recall_plot`."""
    args = parse_args()
    runner = PrecisionRecallPlotRunner(
        max_predictions=args.max_predictions,
        steps=int(args.steps),
        num_k=args.num_k,
    )
    runner.run(pairwise_output_dir=Path(args.pairwise_output_dir))


if __name__ == "__main__":
    main()


