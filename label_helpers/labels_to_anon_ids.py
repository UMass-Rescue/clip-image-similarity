from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_labels(labels_path: Path) -> Dict[str, List[str]]:
    """Load labels JSON (series -> list of image paths).

    Args:
        labels_path: Path to labels JSON file.
    Returns:
        Dict mapping series name to list of image paths.
    Raises:
        ValueError: If the JSON structure or element types are invalid.
    """
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Labels file must be a JSON object of {series: [paths]}.")

    labels: Dict[str, List[str]] = {}
    for series, paths in data.items():
        if not isinstance(series, str):
            raise ValueError("Series names must be strings.")
        if not isinstance(paths, list):
            raise ValueError(f"Labels for series '{series}' must be a list.")
        cleaned: List[str] = []
        for idx, path in enumerate(paths):
            if not isinstance(path, str):
                raise ValueError(
                    f"Labels for series '{series}' must be strings; found {type(path).__name__} at index {idx}."
                )
            cleaned.append(path)
        labels[series] = cleaned
    return labels


def load_anon_map(anon_map_path: Path) -> Dict[str, str]:
    """Load anon map {absolute_path: anon_id}.

    Args:
        anon_map_path: Path to anon map JSON created by the pairwise run.
    Returns:
        Dict mapping absolute path string to anonymous ID string.
    Raises:
        ValueError: If the JSON structure or value types are invalid.
    """
    with anon_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Anon map must be a JSON object of {path: id}.")
    mapping: Dict[str, str] = {}
    for path_str, anon_id in data.items():
        if not isinstance(path_str, str) or not isinstance(anon_id, str):
            raise ValueError("Anon map must use string keys and string values.")
        mapping[path_str] = anon_id
    return mapping


def relabel_to_anon_ids(
    labels: Dict[str, List[str]], anon_map: Dict[str, str]
) -> Dict[str, List[str]]:
    """Convert labels using absolute paths into labels using anon IDs.

    Args:
        labels: Mapping series -> list of absolute paths.
        anon_map: Mapping absolute path -> anon ID.
    Returns:
        Mapping series -> list of anon IDs.
    Raises:
        ValueError: If any path from labels is missing in the anon map.
    """
    result: Dict[str, List[str]] = {}
    missing_paths: List[str] = []

    for series, paths in labels.items():
        anon_ids: List[str] = []
        for path in paths:
            if path not in anon_map:
                missing_paths.append(path)
            else:
                anon_ids.append(anon_map[path])
        result[series] = anon_ids

    if missing_paths:
        preview = "\n  ".join(missing_paths[:20])
        more = "" if len(missing_paths) <= 20 else f"\n  ... and {len(missing_paths) - 20} more"
        raise ValueError(
            "Some label paths are missing from the anon map. Ensure you produced labels against the same files as the pairwise run:\n  "
            f"{preview}{more}"
        )

    return result


def save_labels(labels: Dict[str, List[str]], output_path: Path) -> None:
    """Write labels JSON with anon IDs.

    Args:
        labels: Mapping series -> list of anon IDs.
        output_path: Path to write the JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for labels->anon conversion."""
    parser = argparse.ArgumentParser(
        description="Convert labels JSON with absolute paths to labels using anonymous IDs."
    )
    parser.add_argument("--labels", required=True, help="Path to labels JSON (series -> list of absolute paths).")
    parser.add_argument(
        "--anon-map",
        required=True,
        help="Path to anon ID map JSON produced by the pairwise run (absolute path -> anon id).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the converted labels JSON (series -> list of anon ids).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: load labels and anon map, convert, and save."""
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    anon_map_path = Path(args.anon_map).resolve()
    output_path = Path(args.output).resolve()

    labels = load_labels(labels_path)
    anon_map = load_anon_map(anon_map_path)
    converted = relabel_to_anon_ids(labels, anon_map)
    save_labels(converted, output_path)
    print(f"Wrote anon labels to {output_path}")


if __name__ == "__main__":
    main()
