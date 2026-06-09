from __future__ import annotations

import json
from pathlib import Path
from typing import Collection, Dict, List, Set


def load_label_mapping(labels_path: Path) -> Dict[str, List[str]]:
    """Load and validate a labels JSON file as {series: [path, ...]}."""
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Labels file must be a JSON object of {series: [paths]}.")

    labels: Dict[str, List[str]] = {}
    for series, series_paths in data.items():
        if not isinstance(series, str):
            raise ValueError("Series names must be strings.")
        if not isinstance(series_paths, list):
            raise ValueError(f"Labels for series '{series}' must be a list.")
        paths: List[str] = []
        for idx, path in enumerate(series_paths):
            if not isinstance(path, str):
                raise ValueError(
                    f"Labels for series '{series}' must be strings; found {type(path).__name__} at index {idx}."
                )
            paths.append(path)
        labels[series] = paths
    return labels


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
    data = load_label_mapping(labels_path)

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


def filter_label_mapping_to_available_paths(
    labels: Dict[str, List[str]],
    image_paths: Collection[Path],
    *,
    min_images_per_series: int = 2,
) -> Dict[str, List[str]]:
    """Return labels containing only paths in image_paths and large enough series."""
    available = {p.resolve().as_posix() for p in image_paths}
    filtered: Dict[str, List[str]] = {}

    for series, paths in labels.items():
        retained: List[str] = []
        seen: Set[str] = set()
        for path in paths:
            canonical = Path(path).resolve().as_posix()
            if canonical not in available or canonical in seen:
                continue
            retained.append(canonical)
            seen.add(canonical)
        if len(retained) >= min_images_per_series:
            filtered[series] = retained
    return filtered


def map_label_mapping_to_indices(
    labels: Dict[str, List[str]], image_paths: List[Path]
) -> Dict[str, List[int]]:
    """Convert a validated labels mapping to indices matching image_paths order."""
    index_map = {p.resolve().as_posix(): idx for idx, p in enumerate(image_paths)}
    series_indices: Dict[str, List[int]] = {}
    missing: List[str] = []

    for series, paths in labels.items():
        indices: List[int] = []
        for path in paths:
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
    return map_label_mapping_to_indices(load_label_mapping(labels_path), image_paths)
