from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from clip_image_similarity.serialization import save_json
from clip_image_similarity.utils import log, plural


SeriesMapping = Dict[str, List[object]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for splitting a series JSON file."""
    parser = argparse.ArgumentParser(
        description="Split a series JSON into train/validation(/test) files by series."
    )
    parser.add_argument(
        "--series-json",
        required=True,
        help="Input JSON mapping series labels to image identifiers.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where split JSON files will be written.",
    )
    parser.add_argument(
        "--train-pct",
        required=True,
        type=int,
        help="Integer percentage of series assigned to the training split.",
    )
    parser.add_argument(
        "--val-pct",
        required=True,
        type=int,
        help="Integer percentage of series assigned to the validation split.",
    )
    parser.add_argument(
        "--test-pct",
        default=None,
        type=int,
        help="Optional integer percentage of series assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional random seed for reproducible splits.",
    )
    return parser.parse_args()


def load_series_json(path: Path) -> SeriesMapping:
    """Load and validate a series JSON mapping."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Series JSON must be an object of {series: [items]}.")

    result: SeriesMapping = {}
    for series, items in raw.items():
        if not isinstance(series, str):
            raise ValueError("Series names must be strings.")
        if not isinstance(items, list):
            raise ValueError(f"Series '{series}' must map to a list.")
        result[series] = items
    return result


def split_series_mapping(
    series_mapping: SeriesMapping,
    *,
    train_pct: int,
    val_pct: int,
    test_pct: int | None = None,
    seed: int | None = None,
) -> Dict[str, SeriesMapping]:
    """Randomly split whole series labels into train/val(/test) mappings."""
    split_pcts = _validate_percentages(
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
    )
    split_names = list(split_pcts)
    counts = _allocate_counts(len(series_mapping), list(split_pcts.values()))

    series_names = list(series_mapping)
    random.Random(seed).shuffle(series_names)

    splits: Dict[str, SeriesMapping] = {name: {} for name in split_names}
    start = 0
    for split_name, count in zip(split_names, counts):
        selected = series_names[start : start + count]
        splits[split_name] = {name: series_mapping[name] for name in selected}
        start += count
    return splits


def write_series_splits(
    *,
    series_json: Path,
    out_dir: Path,
    train_pct: int,
    val_pct: int,
    test_pct: int | None = None,
    seed: int | None = None,
) -> Path:
    """Load, split, and write train/val(/test) series JSON files."""
    series_mapping = load_series_json(series_json)
    splits = split_series_mapping(
        series_mapping,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(splits["train"], out_dir / "train_series.json")
    save_json(splits["val"], out_dir / "val_series.json")
    if "test" in splits:
        save_json(splits["test"], out_dir / "test_series.json")

    _log_summary(splits=splits, out_dir=out_dir)
    return out_dir


def _validate_percentages(
    *, train_pct: int, val_pct: int, test_pct: int | None
) -> Dict[str, int]:
    split_pcts = {"train": train_pct, "val": val_pct}
    if test_pct is not None:
        split_pcts["test"] = test_pct

    for split_name, pct in split_pcts.items():
        if pct < 0:
            raise ValueError(f"{split_name}_pct must be non-negative.")

    total = sum(split_pcts.values())
    if total != 100:
        names = ", ".join(f"{name}_pct={pct}" for name, pct in split_pcts.items())
        raise ValueError(f"Split percentages must sum to 100; got {names}.")
    return split_pcts


def _allocate_counts(total_count: int, percentages: Sequence[int]) -> List[int]:
    """Allocate integer split sizes using largest-remainder rounding."""
    raw_counts = [total_count * pct / 100 for pct in percentages]
    counts = [int(value) for value in raw_counts]
    remaining = total_count - sum(counts)
    remainders = sorted(
        range(len(percentages)),
        key=lambda idx: (raw_counts[idx] - counts[idx], percentages[idx]),
        reverse=True,
    )

    for idx in remainders[:remaining]:
        counts[idx] += 1
    return counts


def _log_summary(*, splits: Dict[str, SeriesMapping], out_dir: Path) -> None:
    log("Series split summary:")
    for split_name, mapping in splits.items():
        image_count = sum(len(items) for items in mapping.values())
        log(
            f"  {split_name}: "
            f"{len(mapping)} series, {plural(image_count, 'item')}"
        )
    log(f"  output_dir: {out_dir}")


def main() -> None:
    """Entry point for the standalone series splitting script."""
    args = parse_args()
    write_series_splits(
        series_json=Path(args.series_json).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
