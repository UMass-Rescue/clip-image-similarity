from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_series_indices(path: Path) -> Dict[str, List[int]]:
    """Load a series->indices mapping from JSON.

    Args:
        path: Path to JSON file containing {series_name: [index, ...]}.
    Returns:
        Dict mapping series name -> list of integer indices (may include duplicates; caller may dedupe).
    Raises:
        ValueError: If the JSON structure is invalid.
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


def validate_series_indices(
    series_to_indices: Dict[str, List[int]], n_items: int
) -> Dict[str, List[int]]:
    """Validate and clean series indices to ensure they are in-bounds and unique.

    Args:
        series_to_indices: Mapping series -> list of indices.
        n_items: Total number of items represented in the pairwise distance structure.
    Returns:
        A cleaned mapping with each series' indices deduplicated and sorted.
    Raises:
        ValueError: If any index is out of bounds.
    """
    cleaned: Dict[str, List[int]] = {}
    for series, idxs in series_to_indices.items():
        unique_idxs = sorted(set(idxs))
        for idx in unique_idxs:
            if idx < 0 or idx >= n_items:
                raise ValueError(
                    f"Index {idx} in series '{series}' is out of bounds for distance data of size {n_items}."
                )
        cleaned[series] = unique_idxs
    return cleaned


