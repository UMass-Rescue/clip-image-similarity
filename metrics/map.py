import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ----------------------------
# I/O helpers
# ----------------------------


def load_labels(labels_path: str) -> Dict[str, Set[str]]:
    """Load labels JSON (series -> list of image paths).

    Args:
        labels_path: Path to labels JSON file.
    Returns:
        Dict mapping series name to a set of image paths.
    Raises:
        ValueError: If the structure or element types are invalid.
    """
    with open(labels_path, "r") as f:
        data = json.load(f)
    # Expecting a dict: series_name -> list of image paths
    series_to_images: Dict[str, Set[str]] = {}
    for series, paths in data.items():
        if not isinstance(paths, list):
            raise ValueError(f"Labels for series '{series}' must be a list of paths.")
        normalized: Set[str] = set()
        for idx, path in enumerate(paths):
            if not isinstance(path, str):
                raise ValueError(
                    f"Labels for series '{series}' must be strings; "
                    f"found {type(path).__name__} at index {idx}."
                )
            normalized.add(path)
        series_to_images[series] = normalized
    return series_to_images


def load_pairwise(pairwise_path: str) -> List[dict]:
    """Load pairwise distance results.

    Args:
        pairwise_path: JSON file with list of {image1, image2, distance}.
    Returns:
        List of dicts with validated keys and numeric distance.
    Raises:
        ValueError: If structure or required keys/types are invalid.
    """
    with open(pairwise_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Pairwise results must be a JSON list.")
    normalized_entries: List[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Pairwise entry at index {idx} must be a JSON object.")
        if "image1" not in item or "image2" not in item:
            raise ValueError(
                f"Pairwise entry at index {idx} must have 'image1' and 'image2'."
            )
        a = item.get("image1")
        b = item.get("image2")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(
                f"Pairwise entry at index {idx} must have 'image1' and 'image2' as strings."
            )
        if "distance" not in item:
            raise ValueError(f"Pairwise entry at index {idx} missing 'distance'.")
        d = extract_distance(item.get("distance"))
        normalized_entries.append({"image1": a, "image2": b, "distance": d})
    return normalized_entries


# ----------------------------
# Validation
# ----------------------------


def collect_pairwise_paths(
    entries: List[dict],
) -> Set[str]:
    """
    Collect image paths present in pairwise entries.

    Args:
        entries: Validated pairwise entries.
    Returns:
        Set of all unique image identifiers seen in image1/image2.
    """
    found: Set[str] = set()
    for item in entries:
        found.add(item["image1"])
        found.add(item["image2"])
    return found


def validate_inputs(
    series_to_images: Dict[str, Set[str]],
    pairwise_entries: List[dict],
) -> None:
    """
    Ensure all labeled images are present in the pairwise results.

    Args:
        series_to_images: Mapping of series to labeled image paths.
        pairwise_entries: Pairwise distance entries.
    Raises:
        ValueError: If any labeled image is missing from pairwise results.
    """
    pairwise_paths = collect_pairwise_paths(pairwise_entries)

    missing: List[str] = []
    for series, images in series_to_images.items():
        for p in images:
            if p not in pairwise_paths:
                missing.append(p)
    if missing:
        missing_preview = "\n  ".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            "Validation failed: the following labeled images are not present in the pairwise results:\n  "
            f"{missing_preview}{more}"
        )


# ----------------------------
# Distance handling and rankings
# ----------------------------


def extract_distance(value) -> float:
    """Normalize distance input to a float.

    Args:
        value: Numeric distance or dict containing a numeric 'distance' field.
    Returns:
        Distance as float.
    Raises:
        ValueError: If the value cannot be interpreted as a distance.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if isinstance(value["distance"], (int, float)):
            return float(value["distance"])
    raise ValueError(f"Unsupported distance format: {value!r}")


def build_rankings(
    entries: List[dict],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a neighbor ranking for each image:
    - For each pair (a, b, d), add (b, d) to a's list and (a, d) to b's list.
    - If duplicate pairs occur, keep the smallest distance.
    - Sort neighbor lists by ascending distance.

    Args:
        entries: Pairwise distance entries.
    Returns:
        Dict mapping image -> list of (neighbor, distance) sorted ascending.
    """
    neighbors: Dict[str, Dict[str, float]] = {}

    for item in entries:
        a_raw = item["image1"]
        b_raw = item["image2"]
        d = item["distance"]
        a = a_raw
        b = b_raw

        if a == b:
            continue
        if a not in neighbors:
            neighbors[a] = {}
        if b not in neighbors:
            neighbors[b] = {}

        # Keep the minimum distance if duplicates appear
        if b not in neighbors[a] or d < neighbors[a][b]:
            neighbors[a][b] = d
        if a not in neighbors[b] or d < neighbors[b][a]:
            neighbors[b][a] = d

    # Convert to sorted lists
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for img, nbrs in neighbors.items():
        rankings[img] = sorted(nbrs.items(), key=lambda kv: kv[1])
    return rankings


def filter_entries_to_labeled(entries: List[dict], labeled_set: Set[str]) -> List[dict]:
    """Filter pairwise entries to pairs where both images are labeled and not self-pairs.

    Args:
        entries: List of pairwise distance entries.
        labeled_set: Set of labeled image identifiers.
    Returns:
        Filtered list of entries containing only labeled pairs.
    """
    return [
        e
        for e in entries
        if e["image1"] in labeled_set and e["image2"] in labeled_set and e["image1"] != e["image2"]
    ]


# ----------------------------
# AP / mAP computation
# ----------------------------


def average_precision_at_k(preds: List[str], positives: Set[str], k: int) -> float:
    """
    Compute AP@k for a single query.
    - Denominator = min(k, len(positives))
    - Sum precision at each rank where a positive is found, divided by denominator.

    Args:
        preds: Ranked list of predicted neighbor ids.
        positives: Set of ground truth positives for the query.
        k: Cutoff rank.
    Returns:
        Average precision at k.
    """
    if not positives:
        return 0.0
    denom = min(k, len(positives))
    if denom == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    # Iterate over predictions until we see denom positives or exhaust preds
    for idx, p in enumerate(preds, start=1):
        if p in positives:
            hits += 1
            sum_precisions += hits / idx
            if hits == denom:
                break
        if idx == k:
            # We only care about top-k ranks; continue if k < denom (won't happen since denom<=k)
            break
    return sum_precisions / denom


def mean_average_precision_at_k(
    preds_by_image: Dict[str, List[str]],
    images: Set[str],
    k: int,
) -> float:
    """
    Compute mAP@k over a set of query images, given per-image ranked predictions.

    Args:
        preds_by_image: Mapping of query image -> ranked neighbor list.
        images: Set of images to evaluate.
        k: Cutoff rank.
    Returns:
        Mean average precision at k across all queries.
    """
    if not images:
        raise ValueError("Cannot compute mAP@k for an empty set of images.")

    ap_values: List[float] = []
    for query in images:
        positives = set(images)
        positives.discard(query)
        preds = preds_by_image[query]
        ap = average_precision_at_k(preds, positives, k)
        ap_values.append(ap)
    return sum(ap_values) / len(ap_values)


def compute_series_map(
    series_to_images: Dict[str, Set[str]],
    rankings: Dict[str, List[Tuple[str, float]]],
    max_k: int,
) -> Dict[str, List[float]]:
    """
    For each series, compute mAP@k for k=1..max_k.
    For series with size s, AP uses denom=min(k, s-1) per query.

    Args:
        series_to_images: Mapping of series -> set of images.
        rankings: Neighbor rankings for each image.
        max_k: Maximum k to evaluate.
    Returns:
        Mapping of series -> list of mAP values for k=1..max_k.
    """
    series_to_map: Dict[str, List[float]] = {}
    for series, images in series_to_images.items():
        s = len(images)
        if s == 0:
            raise ValueError(f"Series '{series}' has no images.")
        if s <= 1:
            raise ValueError(f"Series '{series}' has only one image.")
        # Pre-compute predictions per image (neighbor order only)
        preds_by_image: Dict[str, List[str]] = {}
        for img in images:
            ranked = rankings[img]
            preds_by_image[img] = [nbr for (nbr, _) in ranked if nbr != img]

        ap_by_k: List[float] = []
        for k in range(1, max_k + 1):
            series_map_k = mean_average_precision_at_k(preds_by_image, images, k)
            ap_by_k.append(series_map_k)
        series_to_map[series] = ap_by_k
    return series_to_map


# ----------------------------
# Output
# ----------------------------


def write_series_map_csv(
    series_to_map: Dict[str, List[float]], output_csv: str
) -> None:
    """Write per-series and mean mAP values to CSV.

    Args:
        series_to_map: Mapping of series -> list of mAP values.
        output_csv: Destination CSV path.
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Determine header size (max k)
    max_k = max((len(v) for v in series_to_map.values()), default=0)
    fieldnames = ["series"] + [f"map@{k}" for k in range(1, max_k + 1)]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write per-series rows and accumulate for averages
        sums = [0.0 for _ in range(max_k)]
        counts = [0 for _ in range(max_k)]
        for series, values in series_to_map.items():
            row = {"series": series}
            for i, v in enumerate(values, start=1):
                row[f"map@{i}"] = f"{v:.6f}"
                if i - 1 < max_k:
                    sums[i - 1] += v
                    counts[i - 1] += 1
            writer.writerow(row)
        # Append average row
        if max_k > 0 and any(c > 0 for c in counts):
            avg_row = {"series": "mean"}
            for i in range(max_k):
                avg = (sums[i] / counts[i]) if counts[i] > 0 else 0.0
                avg_row[f"map@{i+1}"] = f"{avg:.6f}"
            writer.writerow(avg_row)


# ----------------------------
# Main
# ----------------------------


def main():
    """CLI entry point to compute mAP@k from labels and pairwise distances."""
    parser = argparse.ArgumentParser(
        description="Compute mAP@k from pairwise image distances and series labels."
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels JSON (series -> list of image paths).",
    )
    parser.add_argument(
        "--pairwise",
        required=True,
        help="Path to pairwise results JSON from pairwise_test.py.",
    )
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--labeled-images-only",
        action="store_true",
        help="If set, filter pairwise entries to pairs where both images are labeled (self-pairs dropped).",
    )
    args = parser.parse_args()

    series_to_images = load_labels(args.labels)
    entries = load_pairwise(args.pairwise)
    if args.labeled_images_only:
        labeled_set = set().union(*series_to_images.values())
        original_len = len(entries)
        entries = filter_entries_to_labeled(entries, labeled_set)
        filtered_len = len(entries)
        if filtered_len == 0:
            raise ValueError("After filtering to labeled images only, no pairwise entries remain.")
        print(
            f"Filtered pairwise entries to labeled images only (removed {original_len - filtered_len} of {original_len})."
        )
    validate_inputs(series_to_images, entries)

    # Build rankings and compute mAP
    rankings = build_rankings(entries)
    max_k = max(len(v) for v in series_to_images.values())
    series_to_map = compute_series_map(series_to_images, rankings, max_k)

    write_series_map_csv(series_to_map, args.output_csv)


if __name__ == "__main__":
    main()
