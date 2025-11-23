from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def flatten_upper_triangle(dist_matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the upper triangular part (i<j) of a square distance matrix into a 1D tensor.

    Args:
        dist_matrix: Square tensor of shape (N, N).
    Returns:
        1D tensor of length N*(N-1)/2 containing distances in row-major upper-tri order.
    """
    n = dist_matrix.shape[0]
    if dist_matrix.shape[1] != n:
        raise ValueError("Distance matrix must be square.")
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=dist_matrix.device)
    return dist_matrix[idx_i, idx_j]


def _infer_n_from_length(length: int) -> int:
    """Infer matrix size N from flattened length = N*(N-1)/2."""
    if length <= 0:
        raise ValueError("Flattened length must be positive.")
    n = (1 + math.isqrt(1 + 8 * length)) // 2
    if n * (n - 1) // 2 != length:
        raise ValueError("Flattened length is not compatible with an upper-triangular matrix.")
    return int(n)


class PackedDistances:
    """Helper for accessing distances stored as a flattened upper-triangle array."""

    def __init__(self, flat: np.ndarray):
        self.flat = flat
        self.n = _infer_n_from_length(len(flat))

    def _pair_index(self, i: int, j: int) -> int:
        if i == j:
            raise ValueError("Self distances are not stored.")
        if i > j:
            i, j = j, i
        return i * (self.n - 1) - (i * (i - 1)) // 2 + (j - i - 1)

    def distance(self, i: int, j: int) -> float:
        """Return the distance between indices i and j."""
        idx = self._pair_index(i, j)
        return float(self.flat[idx])

    @classmethod
    def load(cls, path: Path, key: str = "distances") -> PackedDistances:
        """Load a packed distance array from an .npz file."""
        data = np.load(path, allow_pickle=False)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in npz file.")
        return cls(data[key])

    @staticmethod
    def save(flat: np.ndarray, path: Path, key: str = "distances") -> None:
        """Save a packed distance array to an .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: flat}, allow_pickle=False)
