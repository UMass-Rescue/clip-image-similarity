#!/usr/bin/env python3
"""Filter a labels JSON down to images Pillow can decode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from clip_image_similarity.image_loader import load_rgb_image
from clip_image_similarity.labels import load_label_mapping
from clip_image_similarity.serialization import save_json
from clip_image_similarity.utils import log, plural


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove unloadable image paths and too-small series from a labels JSON."
    )
    parser.add_argument(
        "--labels-json",
        required=True,
        help="Input labels JSON mapping series names to image paths.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output labels JSON containing only loadable image paths.",
    )
    parser.add_argument(
        "--skipped-json",
        required=True,
        help="Output manifest describing skipped images and dropped series.",
    )
    parser.add_argument(
        "--min-images-per-series",
        type=int,
        default=2,
        help="Drop series with fewer retained images than this value (default: 2).",
    )
    return parser.parse_args()


def filter_loadable_labels(
    *,
    labels_json: Path,
    output_json: Path,
    skipped_json: Path,
    min_images_per_series: int = 2,
) -> Dict[str, List[str]]:
    """Write a loadable labels JSON and skipped-image manifest."""
    if min_images_per_series <= 0:
        raise ValueError("--min-images-per-series must be positive.")

    labels = load_label_mapping(labels_json)
    ordered_paths: List[str] = []
    affected_series: Dict[str, List[str]] = {}
    for series, paths in labels.items():
        for path in paths:
            canonical = Path(path).resolve().as_posix()
            if canonical not in affected_series:
                ordered_paths.append(canonical)
                affected_series[canonical] = []
            if series not in affected_series[canonical]:
                affected_series[canonical].append(series)

    loadable_paths: set[str] = set()
    skipped_images = []
    for canonical in ordered_paths:
        path = Path(canonical)
        try:
            image = load_rgb_image(path)
            image.close()
        except Exception as exc:
            skipped_images.append(
                {
                    "path": canonical,
                    "affected_series": affected_series[canonical],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        loadable_paths.add(canonical)

    cleaned: Dict[str, List[str]] = {}
    dropped_series = []
    for series, paths in labels.items():
        retained: List[str] = []
        seen: set[str] = set()
        for path in paths:
            canonical = Path(path).resolve().as_posix()
            if canonical not in loadable_paths or canonical in seen:
                continue
            retained.append(canonical)
            seen.add(canonical)
        if len(retained) >= min_images_per_series:
            cleaned[series] = retained
        else:
            dropped_series.append(
                {
                    "series": series,
                    "retained_image_count": len(retained),
                    "min_images_per_series": min_images_per_series,
                }
            )

    manifest = {
        "input_series_count": len(labels),
        "input_unique_image_count": len(ordered_paths),
        "output_series_count": len(cleaned),
        "output_image_reference_count": sum(len(paths) for paths in cleaned.values()),
        "failed_image_count": len(skipped_images),
        "dropped_series_count": len(dropped_series),
        "skipped_images": skipped_images,
        "dropped_series": dropped_series,
    }
    save_json(manifest, skipped_json)

    if not cleaned:
        raise RuntimeError(
            "No series remain after filtering loadable images and applying "
            f"min_images_per_series={min_images_per_series}."
        )

    save_json(cleaned, output_json)
    log(
        "Loadable label filtering retained "
        f"{plural(len(cleaned), 'series')} and skipped "
        f"{plural(len(skipped_images), 'image')}."
    )
    return cleaned


def main() -> int:
    args = parse_args()
    try:
        filter_loadable_labels(
            labels_json=Path(args.labels_json).resolve(),
            output_json=Path(args.output_json).resolve(),
            skipped_json=Path(args.skipped_json).resolve(),
            min_images_per_series=args.min_images_per_series,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
