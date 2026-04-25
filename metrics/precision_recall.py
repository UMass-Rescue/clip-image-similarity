from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Set


def precision(preds: List[int], positives: Set[int]) -> float:
    """Compute precision for a list of predictions against a set of positives.

    Precision is the fraction of predicted items that are relevant:
      precision = |preds ∩ positives| / |preds|

    Args:
        preds: Ranked list of predicted item indices (e.g., top-k nearest neighbors).
        positives: Set of ground-truth positive item indices for the same query.
    Returns:
        Precision value in [0, 1]. Returns 0.0 if preds is empty.
    """
    if not preds:
        return 0.0
    hits = sum(1 for p in preds if p in positives)
    return hits / len(preds)


def recall(preds: List[int], positives: Set[int]) -> float:
    """Compute recall for a list of predictions against a set of positives.

    Recall is the fraction of relevant items that were retrieved:
      recall = |preds ∩ positives| / |positives|

    Args:
        preds: Ranked list of predicted item indices (e.g., top-k nearest neighbors).
        positives: Set of ground-truth positive item indices for the same query.
    Returns:
        Recall value in [0, 1] when preds has no duplicates. Returns 0.0 if positives is empty.
    """
    if not positives:
        return 0.0
    hits = sum(1 for p in preds if p in positives)
    return hits / len(positives)


@dataclass(frozen=True)
class MetricAtK:
    """Aggregate metric summary at a specific k.

    Args:
        value: Metric value (hits / denom).
        hits: Numerator of the metric.
        denom: Denominator of the metric.
    """

    value: float
    hits: int
    denom: int


@dataclass(frozen=True)
class SeriesPrecisionRecallAtK:
    """Precision@k and Recall@k aggregated over a series.

    Args:
        k: Number of predictions considered per query (top-k).
        precision: Aggregated precision at k.
        recall: Aggregated recall at k.
        hits_by_query: Per-query hit counts at k (numerator used by both precision and recall).
    """

    k: int
    precision: MetricAtK
    recall: MetricAtK
    hits_by_query: Dict[int, int]


def _hits(preds: Sequence[int], positives: Set[int]) -> int:
    """Count how many predictions are positives.

    Args:
        preds: Predicted indices (assumed unique).
        positives: Set of positive indices.
    Returns:
        Number of hits in preds.
    """
    return sum(1 for p in preds if p in positives)


def series_precision_recall_at_k(
    preds_by_query: Dict[int, List[int]],
    series_indices: Sequence[int],
    max_predictions: int,
    *,
    ks: Sequence[int] | None = None,
) -> List[SeriesPrecisionRecallAtK]:
    """Compute aggregated Precision@k and Recall@k curves for a series.

    This treats every index in the series as a query. For query q, positives are all
    other indices in the series (series_set - {q}). For each k in [1, max_predictions],
    we compute:
      - total_hits(k): sum of hits in preds_by_query[q][:k] across all queries
      - precision@k = total_hits(k) / total_predictions_considered(k)
      - recall@k = total_hits(k) / total_positives_considered

    Args:
        preds_by_query: Mapping query index -> ranked list of prediction indices.
            Assumes each list is ordered (nearest first) and contains unique indices.
        series_indices: Indices belonging to the series.
        max_predictions: Maximum k to compute up to (inclusive). Must be <= number of
            predictions available for each query after removing the query index itself.
        ks: Optional explicit list of k values to evaluate. If provided, each k must be
            in [1, max_predictions]. The returned curve will contain exactly these k values.
    Returns:
        A list of SeriesPrecisionRecallAtK, one per selected k.
    Raises:
        ValueError: If max_predictions exceeds available predictions, or if prediction
            lists are not all the same length, or if a series query is missing.
    """
    series = sorted(set(int(x) for x in series_indices))
    series_set = set(series)

    if not series:
        return []

    for q in series:
        if q not in preds_by_query:
            raise ValueError(f"Missing predictions for query index {q}.")

    # Filter predictions so we never allow a query to "retrieve itself", then validate
    # that max_predictions is feasible for every query in the series.
    filtered_preds_by_query: Dict[int, List[int]] = {}
    filtered_lengths: set[int] = set()
    for q in series:
        filtered = [p for p in preds_by_query[q] if p != q]
        filtered_preds_by_query[q] = filtered
        filtered_lengths.add(len(filtered))
    if len(filtered_lengths) != 1:
        raise ValueError(
            "All queries must have the same number of predictions after removing self."
        )
    (n_preds_per_query,) = tuple(filtered_lengths)
    if max_predictions > n_preds_per_query:
        raise ValueError(
            f"max_predictions ({max_predictions}) exceeds available predictions per query ({n_preds_per_query})."
        )

    results: List[SeriesPrecisionRecallAtK] = []
    ks_eval = list(ks) if ks is not None else list(range(1, max_predictions + 1))
    if not ks_eval:
        return []
    if min(ks_eval) < 1 or max(ks_eval) > max_predictions:
        raise ValueError("All k values must be in [1, max_predictions].")
    if sorted(ks_eval) != list(ks_eval):
        raise ValueError("k values must be sorted in ascending order.")

    for k in ks_eval:
        hits_by_query: Dict[int, int] = {}
        total_hits = 0
        total_predictions_considered = 0
        total_positives_considered = 0

        for q in series:
            positives = set(series_set)
            positives.discard(q)
            preds_k = filtered_preds_by_query[q][:k]
            hits = _hits(preds_k, positives)

            hits_by_query[q] = hits
            total_hits += hits
            total_predictions_considered += len(preds_k)
            total_positives_considered += len(positives)

        precision_value = (
            total_hits / total_predictions_considered
            if total_predictions_considered > 0
            else 0.0
        )
        recall_value = (
            total_hits / total_positives_considered
            if total_positives_considered > 0
            else 0.0
        )
        results.append(
            SeriesPrecisionRecallAtK(
                k=k,
                precision=MetricAtK(
                    value=float(precision_value),
                    hits=int(total_hits),
                    denom=int(total_predictions_considered),
                ),
                recall=MetricAtK(
                    value=float(recall_value),
                    hits=int(total_hits),
                    denom=int(total_positives_considered),
                ),
                hits_by_query=hits_by_query,
            )
        )
    return results


@dataclass(frozen=True)
class DatasetPrecisionRecallAtK:
    """Precision@k and Recall@k aggregated across a dataset.

    Args:
        k: Number of predictions considered per query (top-k).
        precision: Dataset-level precision at k.
        recall: Dataset-level recall at k.
    """

    k: int
    precision: MetricAtK
    recall: MetricAtK


@dataclass(frozen=True)
class DatasetPrecisionRecallResult:
    """Dataset-level PR@k curve plus per-series curves.

    Args:
        dataset_curve: Dataset-level precision/recall for k=1..max_predictions.
        series_curves: Mapping series name -> per-series PR@k curve.
    """

    dataset_curve: List[DatasetPrecisionRecallAtK]
    series_curves: Dict[str, List[SeriesPrecisionRecallAtK]]


def select_k_values(max_predictions: int, steps: int = 1) -> List[int]:
    """Select k values between [1, max_predictions] based on a step size.

    Examples:
      - steps=1 -> [1, 2, 3, ..., max_predictions]
      - steps=2 -> [1, 3, 5, ..., max_predictions] (max_predictions always included)

    Args:
        max_predictions: Maximum k value to consider (inclusive).
        steps: Step size for selecting k values. Must be >= 1.
    Returns:
        Sorted list of k values, always including max_predictions (if max_predictions >= 1).
    Raises:
        ValueError: If max_predictions < 1 or steps < 1.
    """
    if max_predictions < 1:
        raise ValueError("max_predictions must be >= 1.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    ks = list(range(1, max_predictions + 1, steps))
    if ks[-1] != max_predictions:
        ks.append(max_predictions)
    return ks


def select_k_values_log(max_predictions: int, num_k: int) -> List[int]:
    """Select k values between [1, max_predictions] in (approximately) log space.

    This produces more k samples for small k, and fewer samples as k grows.

    Args:
        max_predictions: Maximum k value to consider (inclusive).
        num_k: Target number of k values to evaluate. Must be >= 2 when max_predictions > 1.
    Returns:
        Sorted, unique list of k values, always including 1 and max_predictions.
    Raises:
        ValueError: If max_predictions < 1, or num_k is invalid.
    """
    if max_predictions < 1:
        raise ValueError("max_predictions must be >= 1.")
    if max_predictions == 1:
        return [1]
    if num_k < 2:
        raise ValueError("num_k must be >= 2 when max_predictions > 1.")

    # Generate log-spaced values in [1, max_predictions], then round to ints.
    log_k = math.log(max_predictions)
    raw: List[int] = []
    for i in range(num_k):
        t = i / (num_k - 1)
        k = int(round(math.exp(t * log_k)))
        raw.append(max(1, min(max_predictions, k)))

    # Deduplicate and ensure endpoints.
    ks = sorted(set(raw + [1, max_predictions]))
    return ks


def dataset_precision_recall_at_k(
    preds_by_query: Dict[int, List[int]],
    series_to_indices: Dict[str, Sequence[int]],
    max_predictions: int,
    *,
    ks: Sequence[int] | None = None,
    show_progress: bool = False,
) -> DatasetPrecisionRecallResult:
    """Compute dataset-level Precision@k and Recall@k by aggregating over all series.

    This function computes a PR@k curve per series using `series_precision_recall_at_k`,
    then aggregates totals across all series at each k.

    Aggregation is **micro-averaged**:
      - dataset_precision@k = (sum_series hits@k) / (sum_series precision_denoms@k)
      - dataset_recall@k    = (sum_series hits@k) / (sum_series recall_denoms@k)

    Args:
        preds_by_query: Mapping query index -> ranked list of prediction indices.
        series_to_indices: Mapping series name -> indices belonging to that series.
        max_predictions: Maximum k to compute up to (inclusive).
        ks: Optional explicit list of k values to evaluate. If provided, each k must be
            in [1, max_predictions]. The returned curves will contain exactly these k values.
        show_progress: If True, show high-level progress bars via tqdm.
    Returns:
        DatasetPrecisionRecallResult containing dataset_curve and per-series curves.
    Raises:
        ValueError: If any series computation fails (e.g., max_predictions too large).
    """
    ks_eval = list(ks) if ks is not None else list(range(1, max_predictions + 1))
    if not ks_eval:
        return DatasetPrecisionRecallResult(dataset_curve=[], series_curves={})
    if min(ks_eval) < 1 or max(ks_eval) > max_predictions:
        raise ValueError("All k values must be in [1, max_predictions].")
    if sorted(ks_eval) != list(ks_eval):
        raise ValueError("k values must be sorted in ascending order.")

    series_curves: Dict[str, List[SeriesPrecisionRecallAtK]] = {}
    series_iter = series_to_indices.items()
    if show_progress:
        from tqdm import tqdm  # type: ignore

        series_iter = tqdm(
            list(series_iter),
            desc="Computing per-series PR@k",
            unit="series",
        )
    for series_name, idxs in series_iter:
        series_curves[series_name] = series_precision_recall_at_k(
            preds_by_query=preds_by_query,
            series_indices=idxs,
            max_predictions=max_predictions,
            ks=ks_eval,
        )

    dataset_curve: List[DatasetPrecisionRecallAtK] = []
    k_iter = ks_eval
    if show_progress:
        from tqdm import tqdm  # type: ignore

        k_iter = tqdm(list(k_iter), desc="Aggregating dataset PR@k", unit="k")
    for idx, k in enumerate(k_iter):
        total_hits = 0
        total_precision_denom = 0
        total_recall_denom = 0
        for curve in series_curves.values():
            point = curve[idx]
            if point.k != k:
                raise ValueError("Series PR@k curves are not aligned across series.")
            total_hits += point.precision.hits
            total_precision_denom += point.precision.denom
            total_recall_denom += point.recall.denom

        precision_value = (
            total_hits / total_precision_denom if total_precision_denom > 0 else 0.0
        )
        recall_value = total_hits / total_recall_denom if total_recall_denom > 0 else 0.0
        dataset_curve.append(
            DatasetPrecisionRecallAtK(
                k=k,
                precision=MetricAtK(
                    value=float(precision_value),
                    hits=int(total_hits),
                    denom=int(total_precision_denom),
                ),
                recall=MetricAtK(
                    value=float(recall_value),
                    hits=int(total_hits),
                    denom=int(total_recall_denom),
                ),
            )
        )

    return DatasetPrecisionRecallResult(
        dataset_curve=dataset_curve,
        series_curves=series_curves,
    )


