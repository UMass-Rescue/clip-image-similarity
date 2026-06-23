from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

from clip_image_similarity.packed_distances import PackedDistances
from metrics.accuracy_hits_k import _labeled_union, _resolve_inputs
from metrics.precision_recall import DatasetPrecisionRecallAtK, dataset_precision_recall_at_k
from metrics.precision_recall_plot import build_predictions_for_queries
from metrics.series_indices import load_series_indices, validate_series_indices


def _validate_ks(ks: Sequence[int]) -> List[int]:
    ks_eval = [int(k) for k in ks]
    if not ks_eval:
        return []
    if min(ks_eval) < 1:
        raise ValueError("All k values must be >= 1.")
    if sorted(ks_eval) != ks_eval:
        raise ValueError("k values must be sorted in ascending order.")
    return ks_eval


def _write_csv(rows: Sequence[DatasetPrecisionRecallAtK], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "hits",
                "precision_denom",
                "recall_denom",
                "precision_at_k",
                "recall_at_k",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "k": r.k,
                    "hits": r.precision.hits,
                    "precision_denom": r.precision.denom,
                    "recall_denom": r.recall.denom,
                    "precision_at_k": f"{r.precision.value:.6f}",
                    "recall_at_k": f"{r.recall.value:.6f}",
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level precision@k and recall@k from pairwise distances "
            "and series_to_indices."
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
        required=False,
        help="One or more k values (ascending), e.g. --k 1 5 10.",
    )
    parser.add_argument(
        "--all-k",
        action="store_true",
        help="Evaluate every k from 1 through the maximum available prediction count.",
    )
    parser.add_argument(
        "--output-csv",
        required=False,
        help="Optional CSV path to write k, hits, denominators, precision_at_k, and recall_at_k.",
    )
    parser.add_argument(
        "--labeled-images-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict retrieval candidates to labeled images only (default: True). "
            "Use --no-labeled-images-only to consider all images in the matrix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pairwise_output_dir = Path(args.pairwise_output_dir).resolve()
    if args.all_k and args.k:
        raise ValueError("Provide either --k or --all-k, not both.")
    if not args.all_k and not args.k:
        raise ValueError("Provide --k or --all-k.")
    ks = _validate_ks(args.k or [])

    dist_path, series_path = _resolve_inputs(pairwise_output_dir)
    dist = PackedDistances.load(dist_path)

    series_to_indices = load_series_indices(series_path)
    series_to_indices = validate_series_indices(series_to_indices, dist.n)
    queries = _labeled_union(series_to_indices)

    preds_by_query = build_predictions_for_queries(dist, queries, show_progress=True)
    max_predictions = dist.n - 1

    if args.labeled_images_only:
        labeled_set = set(queries)
        preds_by_query = {
            q: [p for p in preds if p in labeled_set]
            for q, preds in preds_by_query.items()
        }
        max_predictions = max(0, len(labeled_set) - 1)

    if args.all_k:
        ks = list(range(1, max_predictions + 1))

    if ks and max(ks) > max_predictions:
        raise ValueError(
            f"Requested k={max(ks)} exceeds available predictions per query ({max_predictions})."
        )

    result = dataset_precision_recall_at_k(
        preds_by_query=preds_by_query,
        series_to_indices=series_to_indices,
        max_predictions=max_predictions,
        ks=ks,
        show_progress=True,
    )

    print("k,hits,precision_denom,recall_denom,precision_at_k,recall_at_k")
    for r in result.dataset_curve:
        print(
            f"{r.k},{r.precision.hits},{r.precision.denom},{r.recall.denom},"
            f"{r.precision.value:.6f},{r.recall.value:.6f}"
        )

    if args.output_csv:
        _write_csv(result.dataset_curve, Path(args.output_csv).resolve())


if __name__ == "__main__":
    main()
