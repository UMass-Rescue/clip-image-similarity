from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set


def extract_labeled_paths(labels_path: Path) -> List[Path]:
    """Return sorted unique image paths referenced in a labels JSON file.

    Args:
        labels_path: Path to labels JSON (series -> list of image paths).
    Returns:
        Sorted list of resolved, deduplicated image Paths.
    Raises:
        FileNotFoundError: If any labeled image path does not exist on disk.
        ValueError: If the labels file is malformed.
    """
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Labels file must be a JSON object of {series: [paths]}.")

    seen: Set[str] = set()
    paths: List[Path] = []
    for series, series_paths in data.items():
        if not isinstance(series_paths, list):
            raise ValueError(f"Labels for series '{series}' must be a list.")
        for p in series_paths:
            resolved = Path(p).resolve()
            key = resolved.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if not resolved.is_file():
                raise FileNotFoundError(f"Labeled image not found: {resolved}")
            paths.append(resolved)
    return sorted(paths)


def map_labels_to_indices(
    labels_path: Path, image_paths: List[Path]
) -> Dict[str, List[int]]:
    """Convert labels (series -> image paths) to indices matching image_paths order.

    Args:
        labels_path: Path to labels JSON file (series -> list of image paths).
        image_paths: Ordered list of image paths used for embedding.
    Returns:
        Dict mapping series -> list of integer indices.
    Raises:
        ValueError: If labels are malformed or refer to paths not present in image_paths.
    """
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Labels file must be a JSON object of {series: [paths]}.")

    index_map = {p.resolve().as_posix(): idx for idx, p in enumerate(image_paths)}
    series_indices: Dict[str, List[int]] = {}
    missing: List[str] = []

    for series, paths in data.items():
        if not isinstance(series, str):
            raise ValueError("Series names must be strings.")
        if not isinstance(paths, list):
            raise ValueError(f"Labels for series '{series}' must be a list.")
        indices: List[int] = []
        for idx, path in enumerate(paths):
            if not isinstance(path, str):
                raise ValueError(
                    f"Labels for series '{series}' must be strings; found {type(path).__name__} at index {idx}."
                )
            canonical = Path(path).resolve().as_posix()
            if canonical not in index_map:
                missing.append(path)
            else:
                indices.append(index_map[canonical])
        series_indices[series] = indices

    if missing:
        preview = "\n  ".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            "The following labeled paths were not found in the embedded image list:\n  "
            f"{preview}{more}"
        )

    return series_indices
