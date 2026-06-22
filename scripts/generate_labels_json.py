#!/usr/bin/env python3
"""Generate a series labels JSON from a directory of series folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a labels JSON mapping each immediate child directory to "
            "the image files inside it."
        )
    )
    parser.add_argument(
        "--series-dir",
        required=True,
        help="Directory containing one child directory per series.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path where the generated labels JSON should be written.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help=(
            "Image filename extensions to include, case-insensitive "
            f"(default: {' '.join(DEFAULT_EXTENSIONS)})."
        ),
    )
    parser.add_argument(
        "--min-images-per-series",
        type=int,
        default=1,
        help="Skip series folders with fewer images than this value (default: 1).",
    )
    parser.add_argument(
        "--relative-to",
        help=(
            "Write paths relative to this directory instead of absolute paths. "
            "For the fine-tune workflow, absolute paths are usually simplest."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output JSON.",
    )
    return parser.parse_args()


def generate_labels_json(
    *,
    series_dir: Path,
    output_json: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    min_images_per_series: int = 1,
    relative_to: Path | None = None,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    if not series_dir.is_dir():
        raise NotADirectoryError(f"series-dir does not exist or is not a directory: {series_dir}")
    if min_images_per_series <= 0:
        raise ValueError("--min-images-per-series must be positive.")
    if output_json.exists() and not overwrite:
        raise FileExistsError(f"Output JSON already exists: {output_json}")

    normalized_extensions = normalize_extensions(extensions)
    base = relative_to.resolve() if relative_to is not None else None
    if base is not None and not base.is_dir():
        raise NotADirectoryError(f"relative-to is not a directory: {base}")

    labels: Dict[str, List[str]] = {}
    skipped = []
    for child in sorted(series_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        image_paths = [
            format_path(path.resolve(), base)
            for path in sorted(child.iterdir(), key=lambda path: path.name)
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ]
        if len(image_paths) < min_images_per_series:
            skipped.append(child.name)
            continue
        labels[child.name] = image_paths

    if not labels:
        raise RuntimeError("No series with matching image files were found.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
        f.write("\n")

    image_count = sum(len(paths) for paths in labels.values())
    print(
        f"Wrote {output_json} with {len(labels)} series and "
        f"{image_count} image references."
    )
    if skipped:
        print(
            f"Skipped {len(skipped)} series below min_images_per_series="
            f"{min_images_per_series}."
        )
    return labels


def normalize_extensions(extensions: Iterable[str]) -> set[str]:
    normalized = set()
    for extension in extensions:
        extension = extension.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension)
    if not normalized:
        raise ValueError("At least one image extension must be provided.")
    return normalized


def format_path(path: Path, base: Path | None) -> str:
    if base is None:
        return path.as_posix()
    try:
        return path.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image path is not under relative-to directory: {path}") from exc


def main() -> int:
    args = parse_args()
    try:
        generate_labels_json(
            series_dir=Path(args.series_dir).expanduser().resolve(),
            output_json=Path(args.output_json).expanduser().resolve(),
            extensions=args.extensions,
            min_images_per_series=args.min_images_per_series,
            relative_to=(
                Path(args.relative_to).expanduser().resolve()
                if args.relative_to is not None
                else None
            ),
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
