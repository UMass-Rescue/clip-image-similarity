import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from clip_image_similarity.labels import map_labels_to_indices
from clip_image_similarity.packed_distances import PackedDistances
from clip_image_similarity.topk import TopKNeighbors  # placeholder for top-k format


def load_series_indices(path: Path) -> Dict[str, List[int]]:
    """Load a series->indices mapping from JSON.

    Args:
        path: Path to JSON file.
    Returns:
        Dict mapping series -> list of indices.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("series_indices must be a JSON object of {series: [indices]}.")
    result: Dict[str, List[int]] = {}
    for series, idx_list in data.items():
        if not isinstance(series, str):
            raise ValueError("Series names must be strings.")
        if not isinstance(idx_list, list):
            raise ValueError(f"Indices for series '{series}' must be a list.")
        for idx in idx_list:
            if not isinstance(idx, int):
                raise ValueError(f"Indices for series '{series}' must be integers.")
        result[series] = idx_list
    return result


def build_rankings(
    dist: PackedDistances, candidates: Set[int], queries: Set[int]
) -> Dict[int, List[int]]:
    """Build neighbor rankings (by index) for each query index.

    Args:
        dist: PackedDistances accessor.
        candidates: Candidate indices to consider.
        queries: Query indices to build rankings for.
    Returns:
        Dict mapping query -> list of neighbor indices sorted by distance.
    """
    rankings: Dict[int, List[int]] = {}
    for q in queries:
        neighbors = []
        for j in candidates:
            if j == q:
                continue
            neighbors.append((j, dist.distance(q, j)))
        neighbors.sort(key=lambda kv: kv[1])
        rankings[q] = [idx for idx, _ in neighbors]
    return rankings


def average_precision_at_k(preds: List[int], positives: Set[int], k: int) -> float:
    """Compute AP@k for a single query."""
    if not positives:
        return 0.0
    denom = min(k, len(positives))
    hits = 0
    sum_precisions = 0.0
    for idx, p in enumerate(preds, start=1):
        if p in positives:
            hits += 1
            sum_precisions += hits / idx
            if hits == denom:
                break
        if idx == k:
            break
    return sum_precisions / denom


def mean_average_precision_at_k(
    preds_by_query: Dict[int, List[int]],
    queries: Set[int],
    k: int,
    positives_lookup: Dict[int, Set[int]],
) -> float:
    """Compute mAP@k over a set of queries."""
    values: List[float] = []
    for q in queries:
        ap = average_precision_at_k(preds_by_query[q], positives_lookup[q], k)
        values.append(ap)
    return sum(values) / len(values) if values else 0.0


def compute_series_map(
    series_to_indices: Dict[str, List[int]],
    rankings: Dict[int, List[int]],
    max_k: int,
) -> Dict[str, List[float]]:
    """Compute mAP@k for each series given rankings over candidate set.

    Args:
        series_to_indices: Mapping series -> list of indices in the matrix.
        rankings: Neighbor rankings for each query index.
        max_k: Maximum k to evaluate.
    Returns:
        Mapping series -> list of mAP values for k=1..max_k.
    """
    series_map: Dict[str, List[float]] = {}
    for series, imgs in series_to_indices.items():
        images = set(imgs)
        if len(images) <= 1:
            raise ValueError(f"Series '{series}' must have at least two images.")
        positives_lookup: Dict[int, Set[int]] = {}
        for q in images:
            pos = set(images)
            pos.discard(q)
            positives_lookup[q] = pos

        ap_by_k: List[float] = []
        for k in range(1, max_k + 1):
            map_k = mean_average_precision_at_k(rankings, images, k, positives_lookup)
            ap_by_k.append(map_k)
        series_map[series] = ap_by_k
    return series_map


def write_series_map_csv(series_to_map: Dict[str, List[float]], output_csv: Path) -> None:
    """Write per-series and mean mAP values to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    max_k = max((len(v) for v in series_to_map.values()), default=0)
    fieldnames = ["series"] + [f"map@{k}" for k in range(1, max_k + 1)]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        sums = [0.0 for _ in range(max_k)]
        counts = [0 for _ in range(max_k)]
        for series, values in series_to_map.items():
            row = {"series": series}
            for i, v in enumerate(values, start=1):
                row[f"map@{i}"] = f"{v:.6f}"
                sums[i - 1] += v
                counts[i - 1] += 1
            writer.writerow(row)
        if max_k > 0:
            avg_row = {"series": "mean"}
            for i in range(max_k):
                avg = (sums[i] / counts[i]) if counts[i] else 0.0
                avg_row[f"map@{i+1}"] = f"{avg:.6f}"
            writer.writerow(avg_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mAP@k from flattened pairwise distances and labels.")
    parser.add_argument(
        "--distances",
        required=False,
        help="Path to npz file containing flattened distances (upper-tri). Required unless --topk is provided.",
    )
    parser.add_argument(
        "--topk",
        required=False,
        help="Path to npz file containing top-k neighbors (indices/distances). Required unless --distances is provided.",
    )
    parser.add_argument(
        "--series-indices",
        help="Optional path to JSON mapping series -> list of indices (preferred for privacy).",
    )
    parser.add_argument(
        "--labels",
        help="Labels JSON (series -> list of image paths). Required if --series-indices not provided.",
    )
    parser.add_argument(
        "--image-paths",
        help="Path to image_paths.json produced during embedding run. Required if --labels is used.",
    )
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--labeled-images-only",
        action="store_true",
        help="Restrict candidates to labeled images only when building rankings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.distances:
        dist_path = Path(args.distances).resolve()
        packed = PackedDistances.load(dist_path)
        neighbor_source = packed
    elif args.topk:
        # Placeholder: load top-k structure when implemented
        neighbor_source = TopKNeighbors.load(Path(args.topk).resolve())
    else:
        raise ValueError("Provide either --distances or --topk.")

    if args.series_indices:
        series_to_indices = load_series_indices(Path(args.series_indices).resolve())
    else:
        if not args.labels or not args.image_paths:
            raise ValueError("Provide either --series-indices or both --labels and --image-paths.")
        image_paths = json.loads(Path(args.image_paths).read_text())
        img_paths = [Path(p) for p in image_paths]
        series_to_indices = map_labels_to_indices(Path(args.labels).resolve(), img_paths)

    labeled_union: Set[int] = set()
    for vals in series_to_indices.values():
        labeled_union.update(vals)

    candidates = labeled_union if args.labeled_images_only else set(range(packed.n))
    rankings = build_rankings(packed, candidates, labeled_union)
    max_k = max(len(v) for v in series_to_indices.values())
    series_map = compute_series_map(series_to_indices, rankings, max_k)
    write_series_map_csv(series_map, Path(args.output_csv).resolve())


if __name__ == "__main__":
    main()
