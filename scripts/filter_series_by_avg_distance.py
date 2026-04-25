from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from clip_image_similarity.packed_distances import PackedDistances
from clip_image_similarity.serialization import save_json
from clip_image_similarity.utils import log, plural
from metrics.series_indices import load_series_indices, validate_series_indices


@dataclass(frozen=True)
class FilterStats:
    """Summary statistics for a filtering run."""

    input_series_count: int
    input_labeled_image_count: int
    total_matrix_image_count: int
    removed_by_in_series_threshold_count: int
    removed_by_out_of_series_threshold_count: int
    removed_by_both_thresholds_count: int
    unique_threshold_failed_image_count: int
    removed_due_to_min_series_size_count: int
    retained_image_count: int
    dropped_series_count: int
    retained_series_count: int


def parse_args() -> tuple[Path, float, str, float, str]:
    """Parse CLI arguments for average-distance series filtering."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter images from each series using mean within-series distance and "
            "mean out-of-series distance thresholds."
        )
    )
    parser.add_argument(
        "--pairwise-output-dir",
        required=True,
        help=(
            "CLI output directory containing evaluation_results/pairwise_distances.npz, "
            "series_to_indices.json, and image_paths.json."
        ),
    )
    parser.add_argument(
        "--in-series-threshold",
        required=True,
        help=(
            "Absolute cosine-distance upper bound for the mean within-series distance; "
            "images with mean distance above this value are removed."
        ),
    )
    parser.add_argument(
        "--out-of-series-threshold",
        required=True,
        help=(
            "Absolute cosine-distance lower bound for the mean out-of-series distance; "
            "images with mean distance below this value are removed."
        ),
    )
    args = parser.parse_args()
    return (
        Path(args.pairwise_output_dir).resolve(),
        float(args.in_series_threshold),
        str(args.in_series_threshold),
        float(args.out_of_series_threshold),
        str(args.out_of_series_threshold),
    )


def _resolve_inputs(pairwise_output_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve and validate required input files."""
    dist_path = pairwise_output_dir / "evaluation_results" / "pairwise_distances.npz"
    series_path = pairwise_output_dir / "series_to_indices.json"
    image_paths_path = pairwise_output_dir / "image_paths.json"

    if not dist_path.is_file():
        raise FileNotFoundError(f"Distances file not found: {dist_path}")
    if not series_path.is_file():
        raise FileNotFoundError(f"Series indices file not found: {series_path}")
    if not image_paths_path.is_file():
        raise FileNotFoundError(f"Image paths file not found: {image_paths_path}")
    return dist_path, series_path, image_paths_path


def _load_image_paths(path: Path, expected_count: int) -> List[Path]:
    """Load image paths and ensure they align with the pairwise matrix size."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("image_paths.json must contain a JSON list of file path strings.")
    for idx, value in enumerate(raw):
        if not isinstance(value, str):
            raise ValueError(
                f"image_paths.json entries must be strings; found {type(value).__name__} at index {idx}."
            )

    image_paths = [Path(value).resolve() for value in raw]
    if len(image_paths) != expected_count:
        raise ValueError(
            "image_paths.json length does not match pairwise distances size: "
            f"expected {expected_count}, found {len(image_paths)}."
        )
    return image_paths


def _format_threshold_for_dir(threshold: float, threshold_label: str | None = None) -> str:
    """Convert a threshold to a deterministic directory-safe suffix."""
    text = threshold_label.strip() if threshold_label is not None else str(threshold)
    text = text.replace("-", "neg")
    text = text.replace(".", "_")
    return f"t{text}"


def _make_output_dir(
    pairwise_output_dir: Path,
    in_series_threshold: float,
    out_of_series_threshold: float,
    in_series_threshold_label: str | None = None,
    out_of_series_threshold_label: str | None = None,
) -> Path:
    """Create a deterministic output directory for a filter run."""
    in_suffix = _format_threshold_for_dir(
        in_series_threshold, in_series_threshold_label
    )
    out_suffix = _format_threshold_for_dir(
        out_of_series_threshold, out_of_series_threshold_label
    )
    out_dir = (
        pairwise_output_dir
        / f"filtered_series_by_avg_distance_in_{in_suffix}_out_{out_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _mean_within_distances_for_series(
    dist: PackedDistances, series_indices: List[int]
) -> Dict[int, float]:
    """Compute mean within-series distance for each index in a series."""
    if len(series_indices) < 2:
        return {}

    sums = {idx: 0.0 for idx in series_indices}
    counts = {idx: 0 for idx in series_indices}

    for left_pos in range(len(series_indices)):
        left_idx = series_indices[left_pos]
        for right_pos in range(left_pos + 1, len(series_indices)):
            right_idx = series_indices[right_pos]
            value = dist.distance(left_idx, right_idx)
            sums[left_idx] += value
            sums[right_idx] += value
            counts[left_idx] += 1
            counts[right_idx] += 1

    return {idx: sums[idx] / counts[idx] for idx in series_indices}


def _mean_out_of_series_distances_for_series(
    dist: PackedDistances,
    series_indices: List[int],
    n_items: int,
) -> Dict[int, float]:
    """Compute mean out-of-series distance for each index in a series."""
    if not series_indices:
        return {}

    series_set = set(series_indices)
    candidates = [idx for idx in range(n_items) if idx not in series_set]
    if not candidates:
        return {idx: float("inf") for idx in series_indices}

    counts = len(candidates)
    means: Dict[int, float] = {}
    for idx in series_indices:
        total = 0.0
        for candidate in candidates:
            total += dist.distance(idx, candidate)
        means[idx] = total / counts
    return means


def filter_series_to_indices(
    series_to_indices: Dict[str, List[int]],
    dist: PackedDistances,
    in_series_threshold: float,
    out_of_series_threshold: float,
) -> tuple[Dict[str, List[int]], FilterStats]:
    """Filter each series using within-series and out-of-series average distances."""
    filtered: Dict[str, List[int]] = {}
    input_series_count = len(series_to_indices)
    input_labeled_image_count = sum(len(idxs) for idxs in series_to_indices.values())
    removed_by_in_series_threshold_count = 0
    removed_by_out_of_series_threshold_count = 0
    removed_by_both_thresholds_count = 0
    dropped_series_count = 0

    for series_name, idxs in series_to_indices.items():
        if len(idxs) < 2:
            dropped_series_count += 1
            continue

        mean_within = _mean_within_distances_for_series(dist, idxs)
        mean_out = _mean_out_of_series_distances_for_series(dist, idxs, dist.n)

        kept: List[int] = []
        for idx in idxs:
            fails_in = mean_within[idx] > in_series_threshold
            fails_out = mean_out[idx] < out_of_series_threshold
            if fails_in:
                removed_by_in_series_threshold_count += 1
            if fails_out:
                removed_by_out_of_series_threshold_count += 1
            if fails_in and fails_out:
                removed_by_both_thresholds_count += 1
            if not fails_in and not fails_out:
                kept.append(idx)

        if len(kept) >= 2:
            filtered[series_name] = kept
        else:
            dropped_series_count += 1

    retained_image_count = sum(len(idxs) for idxs in filtered.values())
    retained_series_count = len(filtered)
    unique_threshold_failed_image_count = (
        removed_by_in_series_threshold_count
        + removed_by_out_of_series_threshold_count
        - removed_by_both_thresholds_count
    )
    removed_due_to_min_series_size_count = (
        input_labeled_image_count
        - retained_image_count
        - unique_threshold_failed_image_count
    )

    stats = FilterStats(
        input_series_count=input_series_count,
        input_labeled_image_count=input_labeled_image_count,
        total_matrix_image_count=dist.n,
        removed_by_in_series_threshold_count=removed_by_in_series_threshold_count,
        removed_by_out_of_series_threshold_count=removed_by_out_of_series_threshold_count,
        removed_by_both_thresholds_count=removed_by_both_thresholds_count,
        unique_threshold_failed_image_count=unique_threshold_failed_image_count,
        removed_due_to_min_series_size_count=removed_due_to_min_series_size_count,
        retained_image_count=retained_image_count,
        dropped_series_count=dropped_series_count,
        retained_series_count=retained_series_count,
    )
    return filtered, stats


def build_series_to_image_paths(
    series_to_indices: Dict[str, List[int]], image_paths: List[Path]
) -> Dict[str, List[str]]:
    """Convert filtered series indices back to the path-label format used by metrics.map."""
    return {
        series_name: [image_paths[idx].as_posix() for idx in idxs]
        for series_name, idxs in series_to_indices.items()
    }


def _log_summary(
    *,
    in_series_threshold: float,
    out_of_series_threshold: float,
    stats: FilterStats,
    output_dir: Path,
) -> None:
    """Print a concise before/after summary for the filter run."""
    log("Filtering summary:")
    log(
        "  thresholds: "
        f"in-series <= {in_series_threshold}, "
        f"out-of-series >= {out_of_series_threshold}"
    )
    log(
        "  input: "
        f"{plural(stats.input_series_count, 'series')}, "
        f"{plural(stats.input_labeled_image_count, 'labeled image')}, "
        f"{plural(stats.total_matrix_image_count, 'total matrix image')}"
    )
    log(
        "  threshold failures: "
        f"{plural(stats.removed_by_in_series_threshold_count, 'image')} by in-series, "
        f"{plural(stats.removed_by_out_of_series_threshold_count, 'image')} by out-of-series, "
        f"{plural(stats.removed_by_both_thresholds_count, 'image')} by both, "
        f"{plural(stats.unique_threshold_failed_image_count, 'unique image')} removed by thresholds"
    )
    log(
        "  series drop rule: "
        f"{plural(stats.dropped_series_count, 'series')} removed for having fewer than 2 retained images, "
        f"{plural(stats.removed_due_to_min_series_size_count, 'additional image')} removed by that rule"
    )
    log(
        "  output: "
        f"{plural(stats.retained_series_count, 'series')}, "
        f"{plural(stats.retained_image_count, 'labeled image')}"
    )
    log(f"  output_dir: {output_dir}")


def generate_filtered_series_labels(
    pairwise_output_dir: Path,
    in_series_threshold: float,
    out_of_series_threshold: float,
    in_series_threshold_label: str | None = None,
    out_of_series_threshold_label: str | None = None,
) -> Path:
    """Generate filtered series label artifacts under a CLI run directory."""
    pairwise_output_dir = pairwise_output_dir.resolve()
    dist_path, series_path, image_paths_path = _resolve_inputs(pairwise_output_dir)

    log(f"Loading distances from {dist_path}...")
    dist = PackedDistances.load(dist_path)
    log(f"Loaded packed distances for {plural(dist.n, 'image')} (dtype={dist.dtype}).")

    log(f"Loading series indices from {series_path}...")
    series_to_indices = load_series_indices(series_path)
    series_to_indices = validate_series_indices(series_to_indices, dist.n)

    log(f"Loading image paths from {image_paths_path}...")
    image_paths = _load_image_paths(image_paths_path, expected_count=dist.n)

    filtered_series_to_indices, stats = filter_series_to_indices(
        series_to_indices=series_to_indices,
        dist=dist,
        in_series_threshold=in_series_threshold,
        out_of_series_threshold=out_of_series_threshold,
    )
    filtered_series_to_paths = build_series_to_image_paths(
        filtered_series_to_indices, image_paths
    )

    output_dir = _make_output_dir(
        pairwise_output_dir,
        in_series_threshold,
        out_of_series_threshold,
        in_series_threshold_label,
        out_of_series_threshold_label,
    )
    save_json(filtered_series_to_indices, output_dir / "series_to_indices.json")
    save_json(filtered_series_to_paths, output_dir / "series_to_image_paths.json")
    _log_summary(
        in_series_threshold=in_series_threshold,
        out_of_series_threshold=out_of_series_threshold,
        stats=stats,
        output_dir=output_dir,
    )
    return output_dir


def main() -> None:
    """Entry point for the standalone series filtering script."""
    (
        pairwise_output_dir,
        in_series_threshold,
        in_series_threshold_label,
        out_of_series_threshold,
        out_of_series_threshold_label,
    ) = parse_args()
    generate_filtered_series_labels(
        pairwise_output_dir,
        in_series_threshold,
        out_of_series_threshold,
        in_series_threshold_label,
        out_of_series_threshold_label,
    )


if __name__ == "__main__":
    main()
