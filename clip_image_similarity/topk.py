from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def extract_topk_neighbors(
    dist_matrix: torch.Tensor, top_k: int, dtype: str = "float32"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top-k neighbors per row from a distance matrix.

    Args:
        dist_matrix: Square distance matrix (N, N) on CPU/GPU.
        top_k: Number of nearest neighbors to retain per row.
        dtype: Target dtype for stored distances ("float32" or "float16").
    Returns:
        Tuple (indices, distances) each shaped (N, top_k) as numpy arrays.
    Side Effects:
        Distance matrix is modified in place.
    Raises:
        ValueError: If distance matrix is not square or top_k is not positive and less than number of images.
    """
    n = dist_matrix.shape[0]
    if dist_matrix.shape[1] != n:
        raise ValueError("Distance matrix must be square.")
    if top_k <= 0 or top_k >= n:
        raise ValueError("top_k must be positive and less than number of images.")
    target_dtype = torch.float16 if dtype == "float16" else torch.float32

    dist_matrix.fill_diagonal_(torch.inf)

    vals, idxs = torch.topk(dist_matrix, k=top_k, dim=1, largest=False, sorted=True)
    vals = vals.to(target_dtype)

    # Move to CPU numpy
    idx_np = idxs.cpu().numpy()
    index_dtype = np.uint16 if n <= np.iinfo(np.uint16).max + 1 else np.uint32
    indices_np = idx_np.astype(index_dtype, copy=False)
    dist_np = vals.cpu().numpy()
    if dtype == "float16":
        dist_np = dist_np.astype(np.float16, copy=False)
    else:
        dist_np = dist_np.astype(np.float32, copy=False)
    return indices_np, dist_np


def save_topk_neighbors(
    path: Path, indices: np.ndarray, distances: np.ndarray, dtype: str
) -> None:
    """Save top-k neighbor indices/distances to an npz file.

    Args:
        path: Destination path for saving top-k neighbors.
        indices: 2D array of neighbor indices per row (shape N x top_k).
        distances: 2D array of neighbor distances per row (shape N x top_k).
        dtype: String dtype used for distances ("float32" or "float16") for metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    top_k = indices.shape[1]
    np.savez_compressed(
        path,
        indices=indices,
        distances=distances,
        top_k=top_k,
        dtype=dtype,
        index_dtype=str(indices.dtype),
        allow_pickle=False,
    )


class TopKNeighbors:
    """Helper for accessing top-k neighbor structure stored as 2D arrays."""

    def __init__(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        *,
        top_k: int | None = None,
        dtype: str | None = None,
    ):
        if indices.ndim != 2 or distances.ndim != 2:
            raise ValueError("indices and distances must be 2D arrays.")
        if indices.shape != distances.shape:
            raise ValueError(
                f"indices shape {indices.shape} does not match distances shape {distances.shape}."
            )
        if distances.dtype not in (np.float16, np.float32):
            raise ValueError(
                f"Distances dtype must be float16 or float32; found {distances.dtype}."
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"indices dtype must be integer; found {indices.dtype}.")
        k = indices.shape[1]
        if top_k is not None and top_k != k:
            raise ValueError(f"top_k metadata ({top_k}) does not match array width ({k}).")
        self.dtype = dtype or str(distances.dtype)
        self.index_dtype = str(indices.dtype)
        self.indices = indices
        self.distances = distances
        self.n, self.k = indices.shape

    @classmethod
    def load(cls, path: Path) -> "TopKNeighbors":
        """Load top-k neighbors from an npz file."""
        data = np.load(path, allow_pickle=False)
        required_keys = {"indices", "distances", "top_k", "dtype"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(
                f"npz file must contain keys {sorted(required_keys)}; missing {sorted(missing)}."
            )
        neighbors = cls(
            data["indices"],
            data["distances"],
            top_k=int(data["top_k"]),
            dtype=str(data["dtype"]),
        )
        stored_dtype = str(data["dtype"])
        if stored_dtype != neighbors.dtype:
            raise ValueError(
                f"dtype metadata ({stored_dtype}) does not match loaded distances dtype ({neighbors.dtype})."
            )
        if "index_dtype" in data:
            stored_index_dtype = str(data["index_dtype"])
            if stored_index_dtype != neighbors.index_dtype:
                raise ValueError(
                    f"index_dtype metadata ({stored_index_dtype}) does not match loaded dtype ({neighbors.index_dtype})."
                )
        return neighbors

    def neighbors(self, idx: int) -> List[Tuple[int, float]]:
        """
        Return the top-k neighbors for the given index as (neighbor_idx, distance) tuples.

        The returned list is sorted by distance in ascending order (nearest first).

        Args:
            idx: Index of the query item (must be in range [0, n)).
        Returns:
            List of (neighbor_idx, distance) tuples, sorted by distance.
        Raises:
            ValueError: If idx is out of bounds.
        """
        if idx < 0 or idx >= self.n:
            raise ValueError(
                f"Index {idx} is out of bounds for neighbors (valid range: 0 <= idx < {self.n})."
            )
        row_idx = self.indices[idx].tolist()
        row_dist = self.distances[idx].tolist()
        return list(zip(row_idx, row_dist))

    def distance_if_present(self, i: int, j: int) -> float | None:
        """Return the distance if j is among i's stored neighbors; else None.

        Args:
            i: Index of the query item (must be in range [0, n)).
            j: Index of the neighbor to check (must be non-negative).
        Returns:
            Distance from i to j if j is in i's stored neighbors, else None.
        Raises:
            ValueError: If i is out of bounds or j is negative.
        """
        if i < 0 or i >= self.n:
            raise ValueError(
                f"Index i={i} is out of bounds (valid range: 0 <= i < {self.n})."
            )
        if j < 0:
            raise ValueError(f"Index j={j} must be non-negative.")
        row = self.indices[i]
        pos = np.where(row == j)[0]
        if pos.size == 0:
            return None
        return float(self.distances[i, pos[0]])
