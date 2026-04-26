from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from clip_image_similarity.clustering import (
    VALID_LINKAGES,
    build_flat_subseries_mapping,
    build_nested_subseries_mapping,
    cluster_single_series,
    summarize_clustering,
)
from clip_image_similarity.packed_distances import PackedDistances
from clip_image_similarity.serialization import save_json
from clip_image_similarity.utils import log, plural
from metrics.series_indices import load_series_indices, validate_series_indices


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for threshold-based subseries clustering."""
    parser = argparse.ArgumentParser(
        description=(
            "Cluster each original series into distance-thresholded subseries using "
            "saved pairwise distances."
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
        "--distance-threshold",
        required=True,
        type=float,
        help="Maximum linkage distance allowed when merging clusters.",
    )
    parser.add_argument(
        "--linkage",
        choices=VALID_LINKAGES,
        default="average",
        help=(
            "Agglomerative linkage strategy. Average is the default; complete is "
            "stricter, while single can chain through intermediate images."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Drop produced subseries with fewer than this many images (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Defaults to a deterministic subdirectory under "
            "--pairwise-output-dir."
        ),
    )
    return parser.parse_args()


def generate_clustered_series_labels(
    *,
    pairwise_output_dir: Path,
    distance_threshold: float,
    linkage: str = "average",
    min_samples: int = 2,
    output_dir: Path | None = None,
) -> Path:
    """Generate thresholded subseries label artifacts under a run directory."""
    pairwise_output_dir = pairwise_output_dir.resolve()
    dist_path, series_path, image_paths_path = _resolve_inputs(pairwise_output_dir)

    log(f"Loading distances from {dist_path}...")
    distances = PackedDistances.load(dist_path)
    log(
        f"Loaded packed distances for {plural(distances.n, 'image')} "
        f"(dtype={distances.dtype})."
    )

    log(f"Loading series indices from {series_path}...")
    series_to_indices = load_series_indices(series_path)
    series_to_indices = validate_series_indices(series_to_indices, distances.n)
    log(f"Found {plural(len(series_to_indices), 'series')} to cluster.")

    log(f"Loading image paths from {image_paths_path}...")
    image_paths = _load_image_paths(image_paths_path, expected_count=distances.n)

    results = []
    for series_name, indices in tqdm(
        series_to_indices.items(), desc="Clustering series"
    ):
        results.append(
            cluster_single_series(
                series_name=series_name,
                indices=indices,
                distances=distances,
                distance_threshold=distance_threshold,
                linkage=linkage,
                min_samples=min_samples,
            )
        )

    flat_subseries = build_flat_subseries_mapping(results)
    nested_subseries = build_nested_subseries_mapping(results)
    series_to_paths = build_series_to_image_paths(flat_subseries, image_paths)
    summary = summarize_clustering(
        results=results,
        distance_threshold=distance_threshold,
        linkage=linkage,
        min_samples=min_samples,
    )

    out_dir = output_dir.resolve() if output_dir else _make_output_dir(
        pairwise_output_dir=pairwise_output_dir,
        distance_threshold=distance_threshold,
        linkage=linkage,
        min_samples=min_samples,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(flat_subseries, out_dir / "series_to_indices.json")
    save_json(series_to_paths, out_dir / "series_to_image_paths.json")
    save_json(nested_subseries, out_dir / "series_to_subseries_indices.json")
    save_json(asdict(summary), out_dir / "summary.json")
    _log_summary(summary=summary, output_dir=out_dir)
    return out_dir


def build_series_to_image_paths(
    series_to_indices: Dict[str, List[int]], image_paths: List[Path]
) -> Dict[str, List[str]]:
    """Convert flat series indices to the path-label format."""
    return {
        series_name: [image_paths[idx].as_posix() for idx in indices]
        for series_name, indices in series_to_indices.items()
    }


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
                "image_paths.json entries must be strings; "
                f"found {type(value).__name__} at index {idx}."
            )

    image_paths = [Path(value).resolve() for value in raw]
    if len(image_paths) != expected_count:
        raise ValueError(
            "image_paths.json length does not match pairwise distances size: "
            f"expected {expected_count}, found {len(image_paths)}."
        )
    return image_paths


def _make_output_dir(
    *,
    pairwise_output_dir: Path,
    distance_threshold: float,
    linkage: str,
    min_samples: int,
) -> Path:
    """Create the deterministic output directory path for a clustering run."""
    threshold_suffix = _format_threshold_for_dir(distance_threshold)
    return (
        pairwise_output_dir
        / f"clustered_series_by_distance_{threshold_suffix}_{linkage}_min{min_samples}"
    )


def _format_threshold_for_dir(threshold: float) -> str:
    """Convert a threshold to a deterministic directory-safe suffix."""
    text = str(threshold).strip().replace("-", "neg").replace(".", "_")
    return f"t{text}"


def _log_summary(*, summary, output_dir: Path) -> None:
    """Print a concise before/after summary for the clustering run."""
    log("Clustering summary:")
    log(
        "  config: "
        f"distance_threshold={summary.distance_threshold}, "
        f"linkage={summary.linkage}, min_samples={summary.min_samples}"
    )
    log(
        "  input: "
        f"{plural(summary.input_series_count, 'series')}, "
        f"{plural(summary.input_labeled_image_count, 'labeled image')}"
    )
    log(
        "  output: "
        f"{plural(summary.retained_subseries_count, 'subseries')}, "
        f"{plural(summary.retained_image_count, 'image')}"
    )
    log(
        "  dropped: "
        f"{plural(summary.dropped_subseries_count, 'subseries')}, "
        f"{plural(summary.dropped_image_count, 'image')}"
    )
    log(f"  output_dir: {output_dir}")


def main() -> None:
    """Entry point for the standalone subseries clustering script."""
    args = parse_args()
    generate_clustered_series_labels(
        pairwise_output_dir=Path(args.pairwise_output_dir),
        distance_threshold=args.distance_threshold,
        linkage=args.linkage,
        min_samples=args.min_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
