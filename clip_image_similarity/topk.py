from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def extract_topk_neighbors(
    dist_matrix: torch.Tensor, top_k: int, dtype: str = "float32"
) -> Tuple[np.ndarray, np.ndarray]:
    """Placeholder: extract top-k neighbors per row from a distance matrix.

    Args:
        dist_matrix: Square distance matrix (N, N).
        top_k: Number of nearest neighbors to retain per row.
        dtype: Target dtype for stored distances ("float32" or "float16").
    Returns:
        Tuple (indices, distances) each shaped (N, top_k).
    """
    raise NotImplementedError("extract_topk_neighbors is not yet implemented.")


def save_topk_neighbors(path: Path, indices: np.ndarray, distances: np.ndarray, dtype: str) -> None:
    """Placeholder: save top-k neighbor indices/distances to an npz file.

    Args:
        path: Destination path for saving top-k neighbors.
        indices: 2D array of neighbor indices per row (shape N x top_k).
        distances: 2D array of neighbor distances per row (shape N x top_k).
        dtype: String dtype used for distances ("float32" or "float16") for metadata.
    """
    raise NotImplementedError("save_topk_neighbors is not yet implemented.")


class TopKNeighbors:
    """Placeholder helper for accessing top-k neighbor structure stored as 2D arrays."""

    @classmethod
    def load(cls, path: Path) -> "TopKNeighbors":
        """Load top-k neighbors from an npz file."""
        raise NotImplementedError("TopKNeighbors.load is not yet implemented.")

    def neighbors(self, idx: int) -> List[Tuple[int, float]]:
        """Return the top-k neighbors for the given index as (neighbor_idx, distance) tuples."""
        raise NotImplementedError

    def distance_if_present(self, i: int, j: int) -> float | None:
        """Return the distance if j is among i's stored neighbors; else None."""
        raise NotImplementedError
