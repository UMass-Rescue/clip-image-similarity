from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from clip_image_similarity.packed_distances import PackedDistances
from metrics.precision_recall_plot import build_predictions_for_queries
from metrics.series_indices import load_series_indices, validate_series_indices


@dataclass(frozen=True)
class HitsAtK:
    """Hits@k summary for a set of queries.

    Args:
        k: Number of neighbors considered per query.
        value: Accuracy@k value in [0, 1].
        hits: Number of queries with at least one same-series neighbor in top-k.
        denom: Total number of queries evaluated.
    """

    k: int
    value: float
    hits: int
    denom: int


def _validate_ks(ks: Sequence[int]) -> List[int]:
    """Validate and normalize k values."""
    ks_eval = [int(k) for k in ks]
    if not ks_eval:
        return []
    if min(ks_eval) < 1:
        raise ValueError("All k values must be >= 1.")
    if sorted(ks_eval) != ks_eval:
        raise ValueError("k values must be sorted in ascending order.")
    return ks_eval


def _query_has_hit(
    query_idx: int,
    preds: Sequence[int],
    series_key_by_index: Mapping[int, str],
    k: int,
) -> bool:
    """Return True if query has any same-series neighbor in top-k predictions."""
    if query_idx not in series_key_by_index:
        raise ValueError(f"Missing series_key for query index {query_idx}.")

    query_series = series_key_by_index[query_idx]
    filtered_preds = [p for p in preds if p != query_idx]

    if len(filtered_preds) < k:
        raise ValueError(
            f"Query index {query_idx} has only {len(filtered_preds)} predictions after removing self, cannot evaluate k={k}."
        )

    for neighbor_idx in filtered_preds[:k]:
        # Unlabeled neighbors are allowed and treated as non-matches.
        if neighbor_idx not in series_key_by_index:
            continue
        if series_key_by_index[neighbor_idx] == query_series:
            return True

    return False


def accuracy(
    preds_by_query: Dict[int, List[int]],
    series_key_by_index: Mapping[int, str],
    k: int,
) -> float:
    """Compute accuracy@k over all queries using same-series hit criterion.

    A query counts as correct at k if any of its top-k neighbors has the same
    series_key as the query.

    Args:
        preds_by_query: Mapping query index -> ranked neighbor indices.
        series_key_by_index: Mapping image index -> series_key.
        k: Number of top neighbors to consider.
    Returns:
        Accuracy@k in [0, 1]. Returns 0.0 when there are no queries.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")

    query_indices = sorted(preds_by_query.keys())
    if not query_indices:
        return 0.0

    hits = 0
    for query_idx in query_indices:
        if _query_has_hit(
            query_idx=query_idx,
            preds=preds_by_query[query_idx],
            series_key_by_index=series_key_by_index,
            k=k,
        ):
            hits += 1

    return hits / len(query_indices)


def hits_at_k(
    preds_by_query: Dict[int, List[int]],
    series_key_by_index: Mapping[int, str],
    *,
    ks: Sequence[int],
) -> List[HitsAtK]:
    """Compute Hits@k (Accuracy@k) for each requested k.

    Definition:
      hits@k = (# queries with any same-series neighbor in top-k) / (# queries)

    Args:
        preds_by_query: Mapping query index -> ranked neighbor indices.
        series_key_by_index: Mapping image index -> series_key.
        ks: Sorted sequence of k values (each >= 1).
    Returns:
        List of HitsAtK, one per k.
    """
    ks_eval = _validate_ks(ks)
    if not ks_eval:
        return []

    query_indices = sorted(preds_by_query.keys())
    if not query_indices:
        return [HitsAtK(k=k, value=0.0, hits=0, denom=0) for k in ks_eval]

    results: List[HitsAtK] = []
    for k in ks_eval:
        hits = 0
        for query_idx in query_indices:
            if _query_has_hit(
                query_idx=query_idx,
                preds=preds_by_query[query_idx],
                series_key_by_index=series_key_by_index,
                k=k,
            ):
                hits += 1

        denom = len(query_indices)
        results.append(
            HitsAtK(
                k=k,
                value=(hits / denom) if denom > 0 else 0.0,
                hits=hits,
                denom=denom,
            )
        )

    return results


def _resolve_inputs(pairwise_output_dir: Path) -> tuple[Path, Path]:
    """Resolve required files from a clip-image-similarity run output directory."""
    dist_path = pairwise_output_dir / "evaluation_results" / "pairwise_distances.npz"
    series_path = pairwise_output_dir / "series_to_indices.json"
    if not dist_path.is_file():
        raise FileNotFoundError(f"Distances file not found: {dist_path}")
    if not series_path.is_file():
        raise FileNotFoundError(f"Series indices file not found: {series_path}")
    return dist_path, series_path


def _series_key_by_index(series_to_indices: Mapping[str, Sequence[int]]) -> Dict[int, str]:
    """Invert series->indices into index->series_key and reject overlaps."""
    out: Dict[int, str] = {}
    for series_key, idxs in series_to_indices.items():
        for idx in idxs:
            if idx in out and out[idx] != series_key:
                raise ValueError(
                    f"Index {idx} appears in multiple series ({out[idx]!r}, {series_key!r})."
                )
            out[idx] = series_key
    return out


def _labeled_union(series_to_indices: Mapping[str, Sequence[int]]) -> List[int]:
    """Return sorted unique set of labeled image indices across all series."""
    labeled = set()
    for idxs in series_to_indices.values():
        labeled.update(int(i) for i in idxs)
    return sorted(labeled)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute accuracy@k and hits@k from pairwise distances and series_to_indices."
        )
    )
    parser.add_argument(
        "--pairwise-output-dir",
        required=True,
        help="Run output directory containing evaluation_results/pairwise_distances.npz and series_to_indices.json.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        required=True,
        help="One or more k values (ascending), e.g. --k 1 5 10.",
    )
    parser.add_argument(
        "--output-csv",
        required=False,
        help="Optional CSV path to write k, hits, denom, hits_at_k, accuracy_at_k.",
    )
    parser.add_argument(
        "--labeled-images-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict retrieval candidates to labeled images only (default: True). Use --no-labeled-images-only to consider all images in the matrix.",
    )
    return parser.parse_args()


def _write_csv(rows: Sequence[HitsAtK], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "hits", "denom", "hits_at_k", "accuracy_at_k"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "k": r.k,
                    "hits": r.hits,
                    "denom": r.denom,
                    "hits_at_k": f"{r.value:.6f}",
                    "accuracy_at_k": f"{r.value:.6f}",
                }
            )


def main() -> None:
    args = _parse_args()
    pairwise_output_dir = Path(args.pairwise_output_dir).resolve()

    dist_path, series_path = _resolve_inputs(pairwise_output_dir)
    dist = PackedDistances.load(dist_path)

    series_to_indices = load_series_indices(series_path)
    series_to_indices = validate_series_indices(series_to_indices, dist.n)

    queries = _labeled_union(series_to_indices)
    series_map = _series_key_by_index(series_to_indices)

    preds_by_query = build_predictions_for_queries(dist, queries, show_progress=True)

    if args.labeled_images_only:
        labeled_set = set(queries)
        preds_by_query = {
            q: [p for p in preds if p in labeled_set]
            for q, preds in preds_by_query.items()
        }

    rows = hits_at_k(preds_by_query, series_map, ks=args.k)

    print("k,hits,denom,hits_at_k,accuracy_at_k")
    for r in rows:
        print(f"{r.k},{r.hits},{r.denom},{r.value:.6f},{r.value:.6f}")

    if args.output_csv:
        _write_csv(rows, Path(args.output_csv).resolve())


if __name__ == "__main__":
    main()
