from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from clip_image_similarity.packed_distances import PackedDistances
from clip_image_similarity.utils import log, plural
from metrics.series_indices import load_series_indices, validate_series_indices


@dataclass(frozen=True)
class HistogramSpec:
    """Configuration for histogram binning.

    Args:
        bins: Number of equal-width bins.
        range_min: Minimum distance value to include.
        range_max: Maximum distance value to include.
        density: If True, normalize counts to form a probability density.
    """

    bins: int
    range_min: float
    range_max: float
    density: bool

    def edges(self) -> np.ndarray:
        """Return bin edges as a length-(bins+1) array."""
        return np.linspace(self.range_min, self.range_max, self.bins + 1, dtype=np.float64)


def _sanitize_filename_stem(name: str) -> str:
    """Convert an arbitrary series name into a safe filename stem."""
    name = name.strip()
    if not name:
        return "series"
    # Replace path separators and most punctuation with underscores; collapse repeats.
    name = re.sub(r"[\\/]+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "series"


class SeriesDistanceExtractor:
    """Extract within-series and out-of-series distances from packed pairwise distances."""

    def __init__(self, distances: PackedDistances):
        """Create a SeriesDistanceExtractor.

        Args:
            distances: PackedDistances accessor for flattened upper-tri distances.
        """
        self._dist = distances

    def within_series_distances(self, series_indices: Sequence[int]) -> np.ndarray:
        """Compute all unique within-series distances for a series (no duplicates).

        Args:
            series_indices: Indices belonging to the series.
        Returns:
            1D numpy array of within-series distances.
        Raises:
            ValueError: If series has fewer than two unique indices.
        """
        series = sorted(set(int(x) for x in series_indices))
        if len(series) < 2:
            raise ValueError("Series must contain at least two unique indices.")
        values: List[float] = []
        for a in range(len(series)):
            for b in range(a + 1, len(series)):
                values.append(self._dist.distance(series[a], series[b]))
        return np.asarray(values, dtype=np.float32)

    def out_of_series_distances(
        self, series_indices: Sequence[int], candidates: np.ndarray
    ) -> np.ndarray:
        """Compute out-of-series distances for a series.

        Args:
            series_indices: Indices belonging to the series.
            candidates: 1D int array of candidate indices to compare against (must exclude series indices).
        Returns:
            1D numpy array of out-of-series distances.
        """
        series = sorted(set(int(x) for x in series_indices))
        if not series:
            return np.asarray([], dtype=np.float32)
        candidates = np.asarray(candidates, dtype=np.int64).ravel()
        if candidates.size == 0:
            return np.asarray([], dtype=np.float32)
        values: List[float] = []
        for i in series:
            for j in candidates.tolist():
                values.append(self._dist.distance(i, int(j)))
        return np.asarray(values, dtype=np.float32)


def _plot_overlay_histogram_from_values(
    *,
    output_path: Path,
    title: str,
    spec: HistogramSpec,
    within: np.ndarray,
    out: np.ndarray,
    log_y: bool,
) -> None:
    """Write an overlayed histogram plot using matplotlib's histogram implementation.

    Args:
        output_path: Path to write the PNG.
        title: Plot title.
        spec: HistogramSpec controlling bins/range/density.
        within: 1D array of within-series distances.
        out: 1D array of out-of-series distances.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: WPS433

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = spec.edges()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        out,
        bins=edges,
        density=spec.density,
        alpha=0.45,
        color="tab:orange",
        label="out-of-series",
    )
    ax.hist(
        within,
        bins=edges,
        density=spec.density,
        alpha=0.55,
        color="tab:blue",
        label="within-series",
    )
    ax.set_title(title)
    ax.set_xlabel("cosine distance (1 - cosine_similarity)")
    ax.set_ylabel("density" if spec.density else "count")
    ax.legend()
    ax.set_xlim(spec.range_min, spec.range_max)
    if log_y:
        ax.set_yscale("log", nonpositive="clip")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def _make_run_output_dirs(pairwise_output_dir: Path) -> Tuple[Path, Path, Path]:
    """Create and return output directories for a histogram run.

    Args:
        pairwise_output_dir: CLI run output directory (will create subfolders under it).
    Returns:
        Tuple (run_dir, histogram_dir, raw_values_dir).
    """
    graphs_root = pairwise_output_dir / "graphs"
    graphs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = graphs_root / f"distance_histogram_{timestamp}"
    histogram_dir = run_dir / "histogram"
    raw_values_dir = run_dir / "raw_values"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, histogram_dir, raw_values_dir


def _labeled_union(series_to_indices: Dict[str, List[int]]) -> List[int]:
    """Compute the union of all labeled indices across all series."""
    labeled: set[int] = set()
    for idxs in series_to_indices.values():
        labeled.update(int(x) for x in idxs)
    return sorted(labeled)


def _series_out_candidates(
    *,
    n_items: int,
    series_set: set[int],
    labeled_union_idxs: List[int],
    labeled_images_only: bool,
) -> np.ndarray:
    """Build out-of-series candidate indices for a single series.

    Args:
        n_items: Total number of images in the pairwise matrix.
        series_set: Indices in the current series.
        labeled_union_idxs: Sorted labeled union indices.
        labeled_images_only: If True, restrict candidates to labeled images only.
    Returns:
        1D numpy array of candidate indices not in series_set.
    """
    if labeled_images_only:
        return np.asarray(
            [i for i in labeled_union_idxs if i not in series_set], dtype=np.int64
        )

    # Simpler (and slower) on purpose for v1 readability.
    return np.asarray([i for i in range(n_items) if i not in series_set], dtype=np.int64)


class DistanceHistogramRunner:
    """End-to-end runner for computing and saving within/out-of-series distance histograms."""

    def __init__(self, *, spec: HistogramSpec, log_y: bool = True):
        """Create a DistanceHistogramRunner.

        Args:
            spec: HistogramSpec controlling binning and normalization.
            log_y: If True, plot the y-axis on a log scale.
        """
        self._spec = spec
        self._log_y = log_y

    def run(
        self,
        *,
        pairwise_output_dir: Path,
        labeled_images_only: bool,
        save_raw_values: bool,
    ) -> Path:
        """Run histogram generation for all series and overall.

        Args:
            pairwise_output_dir: Output directory produced by `clip_image_similarity.cli`.
            labeled_images_only: If True, restrict out-of-series candidates to labeled union only.
            save_raw_values: If True, save raw values used to plot histograms under raw_values/.
        Returns:
            Path to the created run directory under pairwise_output_dir/graphs/.
        """
        pairwise_output_dir = pairwise_output_dir.resolve()
        dist_path, series_path = _resolve_inputs(pairwise_output_dir)
        run_dir, histogram_dir, raw_values_dir = _make_run_output_dirs(pairwise_output_dir)
        if save_raw_values:
            raw_values_dir.mkdir(parents=True, exist_ok=True)

        log(f"Loading distances from {dist_path}...")
        dist = PackedDistances.load(dist_path)
        log(f"Loaded packed distances for {plural(dist.n, 'image')} (dtype={dist.dtype}).")

        log(f"Loading series indices from {series_path}...")
        series_to_indices = load_series_indices(series_path)
        series_to_indices = validate_series_indices(series_to_indices, dist.n)
        labeled_union_idxs = _labeled_union(series_to_indices)
        log(f"Found {plural(len(series_to_indices), 'series')} and {plural(len(labeled_union_idxs), 'labeled image')}.")

        extractor = SeriesDistanceExtractor(distances=dist)
        overall_within: List[np.ndarray] = []
        overall_out: List[np.ndarray] = []

        for series_name, idxs in series_to_indices.items():
            series_set = set(idxs)
            if len(series_set) < 2:
                raise ValueError(f"Series '{series_name}' must have at least two images.")

            within = extractor.within_series_distances(idxs)

            out_candidates = _series_out_candidates(
                n_items=dist.n,
                series_set=series_set,
                labeled_union_idxs=labeled_union_idxs,
                labeled_images_only=labeled_images_only,
            )
            out_full = extractor.out_of_series_distances(idxs, out_candidates)

            overall_within.append(within)
            overall_out.append(out_full)

            safe_stem = _sanitize_filename_stem(series_name)
            png_path = histogram_dir / f"{safe_stem}.png"
            title = f"{series_name} (within vs out-of-series)"
            _plot_overlay_histogram_from_values(
                output_path=png_path,
                title=title,
                spec=self._spec,
                within=within,
                out=out_full,
                log_y=self._log_y,
            )
            log(f"Wrote histogram: {png_path}")

            if save_raw_values:
                raw_path = raw_values_dir / f"{safe_stem}.npz"
                np.savez_compressed(raw_path, within=within, out=out_full, allow_pickle=False)
                log(f"Wrote raw values: {raw_path}")

        all_png = histogram_dir / "all_series.png"
        within_all = (
            np.concatenate(overall_within, axis=0)
            if overall_within
            else np.asarray([], dtype=np.float32)
        )
        out_all = (
            np.concatenate(overall_out, axis=0)
            if overall_out
            else np.asarray([], dtype=np.float32)
        )
        _plot_overlay_histogram_from_values(
            output_path=all_png,
            title="all_series (within vs out-of-series)",
            spec=self._spec,
            within=within_all,
            out=out_all,
            log_y=self._log_y,
        )
        log(f"Wrote histogram: {all_png}")

        if save_raw_values:
            log(
                "Skipping raw_values/all_series.npz; combine per-series raw values to recreate overall distributions."
            )

        log(f"Done. Outputs saved under {run_dir}")
        return run_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for histogram generation."""
    parser = argparse.ArgumentParser(
        description="Plot histograms of cosine distances: within-series vs out-of-series."
    )
    parser.add_argument(
        "--pairwise-output-dir",
        required=True,
        help="Output directory produced by clip_image_similarity.cli (contains evaluation_results/pairwise_distances.npz and series_to_indices.json).",
    )
    parser.add_argument(
        "--labeled-images-only",
        action="store_true",
        help="Restrict out-of-series candidates to labeled images only (union of series_to_indices.json).",
    )
    parser.add_argument(
        "--save-raw-values",
        action="store_true",
        help="Save raw distance arrays used to plot histograms under graphs/.../raw_values/.",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins.")
    parser.add_argument(
        "--range-min",
        type=float,
        default=0.0,
        help="Minimum distance value to include (default 0.0).",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=2.0,
        help="Maximum distance value to include (default 2.0).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot density-normalized histograms instead of counts.",
    )
    parser.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a log scale on the y-axis (default: enabled). Use --no-log-y to disable.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for `python -m metrics.distance_histogram`."""
    args = parse_args()
    spec = HistogramSpec(
        bins=args.bins,
        range_min=args.range_min,
        range_max=args.range_max,
        density=bool(args.density),
    )
    runner = DistanceHistogramRunner(spec=spec, log_y=bool(args.log_y))
    runner.run(
        pairwise_output_dir=Path(args.pairwise_output_dir),
        labeled_images_only=bool(args.labeled_images_only),
        save_raw_values=bool(args.save_raw_values),
    )


if __name__ == "__main__":
    main()


