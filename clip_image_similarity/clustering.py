from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .packed_distances import PackedDistances


VALID_LINKAGES = ("average", "complete", "single")


@dataclass(frozen=True)
class Subseries:
    """A retained cluster of images from one original series."""

    name: str
    indices: List[int]


@dataclass(frozen=True)
class SeriesClusterResult:
    """Clustering result and summary counts for one original series."""

    original_series: str
    input_image_count: int
    retained_subseries: List[Subseries]
    retained_image_count: int
    dropped_subseries_count: int
    dropped_image_count: int


@dataclass(frozen=True)
class ClusterSummary:
    """Aggregate summary for a full subseries clustering run."""

    distance_threshold: float
    linkage: str
    min_samples: int
    input_series_count: int
    input_labeled_image_count: int
    retained_subseries_count: int
    retained_image_count: int
    dropped_subseries_count: int
    dropped_image_count: int
    series: Dict[str, Dict[str, object]]


def build_precomputed_distance_matrix(
    distances: PackedDistances, indices: Sequence[int]
) -> np.ndarray:
    """Build a square precomputed distance matrix for one series."""
    series_indices = [int(idx) for idx in indices]
    n = len(series_indices)
    matrix = np.zeros((n, n), dtype=np.float32)

    for left_pos in range(n):
        left_idx = series_indices[left_pos]
        for right_pos in range(left_pos + 1, n):
            right_idx = series_indices[right_pos]
            value = distances.distance(left_idx, right_idx)
            matrix[left_pos, right_pos] = value
            matrix[right_pos, left_pos] = value

    return matrix


def cluster_single_series(
    *,
    series_name: str,
    indices: Sequence[int],
    distances: PackedDistances,
    distance_threshold: float,
    linkage: str = "average",
    min_samples: int = 2,
) -> SeriesClusterResult:
    """Cluster one original series into thresholded subseries."""
    _validate_clustering_options(
        distance_threshold=distance_threshold,
        linkage=linkage,
        min_samples=min_samples,
    )

    series_indices = [int(idx) for idx in indices]
    input_image_count = len(series_indices)
    raw_clusters = _cluster_indices(
        series_indices=series_indices,
        distances=distances,
        distance_threshold=distance_threshold,
        linkage=linkage,
    )

    retained_clusters: List[List[int]] = []
    dropped_subseries_count = 0
    dropped_image_count = 0

    for cluster in raw_clusters:
        if len(cluster) < min_samples:
            dropped_subseries_count += 1
            dropped_image_count += len(cluster)
            continue

        retained_clusters.append(cluster)

    retained_subseries = _name_retained_subseries(
        series_name=series_name,
        retained_clusters=retained_clusters,
    )

    retained_image_count = sum(len(sub.indices) for sub in retained_subseries)

    return SeriesClusterResult(
        original_series=series_name,
        input_image_count=input_image_count,
        retained_subseries=retained_subseries,
        retained_image_count=retained_image_count,
        dropped_subseries_count=dropped_subseries_count,
        dropped_image_count=dropped_image_count,
    )


def build_flat_subseries_mapping(
    results: Sequence[SeriesClusterResult],
) -> Dict[str, List[int]]:
    """Convert retained subseries to the downstream-compatible flat label mapping."""
    return {
        subseries.name: subseries.indices
        for result in results
        for subseries in result.retained_subseries
    }


def _name_retained_subseries(
    *, series_name: str, retained_clusters: Sequence[List[int]]
) -> List[Subseries]:
    if len(retained_clusters) == 1:
        return [Subseries(name=series_name, indices=retained_clusters[0])]

    return [
        Subseries(name=f"{series_name}_subseries_{idx}", indices=cluster)
        for idx, cluster in enumerate(retained_clusters)
    ]


def build_nested_subseries_mapping(
    results: Sequence[SeriesClusterResult],
) -> Dict[str, Dict[str, List[int]]]:
    """Convert retained subseries to original-series -> subseries -> indices."""
    return {
        result.original_series: {
            subseries.name: subseries.indices
            for subseries in result.retained_subseries
        }
        for result in results
    }


def summarize_clustering(
    *,
    results: Sequence[SeriesClusterResult],
    distance_threshold: float,
    linkage: str,
    min_samples: int,
) -> ClusterSummary:
    """Build a JSON-serializable summary for a clustering run."""
    input_series_count = len(results)
    input_labeled_image_count = sum(result.input_image_count for result in results)
    retained_subseries_count = sum(
        len(result.retained_subseries) for result in results
    )
    retained_image_count = sum(result.retained_image_count for result in results)
    dropped_subseries_count = sum(
        result.dropped_subseries_count for result in results
    )
    dropped_image_count = sum(result.dropped_image_count for result in results)

    series_summary: Dict[str, Dict[str, object]] = {}
    for result in results:
        series_summary[result.original_series] = {
            "input_image_count": result.input_image_count,
            "retained_subseries_count": len(result.retained_subseries),
            "retained_image_count": result.retained_image_count,
            "dropped_subseries_count": result.dropped_subseries_count,
            "dropped_image_count": result.dropped_image_count,
            "subseries": {
                subseries.name: {
                    "image_count": len(subseries.indices),
                    "indices": subseries.indices,
                }
                for subseries in result.retained_subseries
            },
        }

    return ClusterSummary(
        distance_threshold=distance_threshold,
        linkage=linkage,
        min_samples=min_samples,
        input_series_count=input_series_count,
        input_labeled_image_count=input_labeled_image_count,
        retained_subseries_count=retained_subseries_count,
        retained_image_count=retained_image_count,
        dropped_subseries_count=dropped_subseries_count,
        dropped_image_count=dropped_image_count,
        series=series_summary,
    )


def _cluster_indices(
    *,
    series_indices: List[int],
    distances: PackedDistances,
    distance_threshold: float,
    linkage: str,
) -> List[List[int]]:
    if not series_indices:
        return []
    if len(series_indices) == 1:
        return [series_indices]

    from sklearn.cluster import AgglomerativeClustering

    matrix = build_precomputed_distance_matrix(distances, series_indices)
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="precomputed",
            linkage=linkage,
        )
    labels = clustering.fit_predict(matrix)

    grouped: Dict[int, List[int]] = {}
    for idx, label in zip(series_indices, labels):
        grouped.setdefault(int(label), []).append(idx)

    clusters = [sorted(cluster) for cluster in grouped.values()]
    return sorted(clusters, key=lambda cluster: (cluster[0], len(cluster)))


def _validate_clustering_options(
    *, distance_threshold: float, linkage: str, min_samples: int
) -> None:
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative.")
    if linkage not in VALID_LINKAGES:
        valid = ", ".join(VALID_LINKAGES)
        raise ValueError(f"linkage must be one of: {valid}.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")
