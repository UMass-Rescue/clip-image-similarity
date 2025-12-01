from __future__ import annotations

import argparse
import json
from pathlib import Path

from .labels import map_labels_to_indices
from .serialization import save_json


def parse_args() -> tuple[Path, Path, bool]:
    """Parse CLI arguments for generating anonymous labels.

    Returns:
        Tuple of (output_dir, labels_path, overwrite).
    """
    parser = argparse.ArgumentParser(
        description="Generate anonymous labels (series_to_indices.json) from a previous CLI run."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Path to existing CLI output directory containing image_paths.json.",
    )
    parser.add_argument(
        "--labels",
        "-l",
        required=True,
        help="Path to labels JSON file (series -> list of image paths).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing series_to_indices.json.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    labels_path = Path(args.labels).resolve()
    return output_dir, labels_path, args.overwrite


def generate_anonymous_labels(
    output_dir: Path, labels_path: Path, overwrite: bool = False
) -> None:
    """Generate series_to_indices.json from labels and image_paths.json.

    Args:
        output_dir: Path to existing CLI output directory.
        labels_path: Path to labels JSON file.
        overwrite: Whether to overwrite existing series_to_indices.json.

    Raises:
        FileNotFoundError: If image_paths.json or labels file not found.
        FileExistsError: If series_to_indices.json exists and overwrite is False.
        ValueError: If labels are invalid or refer to missing image paths.
    """
    # Check that image_paths.json exists
    image_paths_file = output_dir / "image_paths.json"
    if not image_paths_file.is_file():
        raise FileNotFoundError(
            f"image_paths.json not found in {output_dir}. "
            "Please ensure this is a valid CLI output directory."
        )

    # Check that labels file exists
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Check if series_to_indices.json already exists
    series_indices_file = output_dir / "series_to_indices.json"
    if series_indices_file.exists() and not overwrite:
        raise FileExistsError(
            f"{series_indices_file} already exists. Use --overwrite to replace it."
        )

    # Load image paths and convert to Path objects
    with image_paths_file.open("r", encoding="utf-8") as f:
        image_path_strings = json.load(f)

    if not isinstance(image_path_strings, list):
        raise ValueError(
            f"image_paths.json must contain a list of paths, got {type(image_path_strings).__name__}"
        )

    image_paths = [Path(p) for p in image_path_strings]

    # Map labels to indices using existing function
    series_indices = map_labels_to_indices(labels_path, image_paths)

    # Save the result
    save_json(series_indices, series_indices_file)
    print(f"Successfully generated {series_indices_file}")
    print(f"Mapped {len(series_indices)} series to indices.")


def main() -> None:
    """Entry point for `python -m clip_image_similarity.generate_anonymous_labels`."""
    output_dir, labels_path, overwrite = parse_args()
    generate_anonymous_labels(output_dir, labels_path, overwrite)


if __name__ == "__main__":
    main()

