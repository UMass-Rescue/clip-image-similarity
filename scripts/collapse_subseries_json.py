from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from clip_image_similarity.serialization import save_json
from clip_image_similarity.utils import log, plural


SeriesMapping = Dict[str, List[object]]
SeriesToSubseriesMapping = Dict[str, Dict[str, List[object]]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for collapsing subseries labels."""
    parser = argparse.ArgumentParser(
        description="Collapse a flat subseries JSON back to original series labels."
    )
    parser.add_argument(
        "--series-json",
        required=True,
        help="Flat series/subseries JSON mapping labels to items.",
    )
    parser.add_argument(
        "--series-to-subseries-json",
        required=True,
        help="Nested JSON mapping original series to subseries labels.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output path for the collapsed series JSON.",
    )
    return parser.parse_args()


def load_series_mapping(path: Path) -> SeriesMapping:
    """Load and validate a flat series mapping."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("series-json must be a JSON object of {series: [items]}.")

    result: SeriesMapping = {}
    for series, items in raw.items():
        if not isinstance(series, str):
            raise ValueError("series-json labels must be strings.")
        if not isinstance(items, list):
            raise ValueError(f"series-json label '{series}' must map to a list.")
        result[series] = items
    return result


def load_series_to_subseries_mapping(path: Path) -> SeriesToSubseriesMapping:
    """Load and validate original-series -> subseries mapping."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            "series-to-subseries-json must be a JSON object of "
            "{series: {subseries: [items]}}."
        )

    result: SeriesToSubseriesMapping = {}
    for series, subseries_mapping in raw.items():
        if not isinstance(series, str):
            raise ValueError("Original series labels must be strings.")
        if not isinstance(subseries_mapping, dict):
            raise ValueError(f"Original series '{series}' must map to an object.")

        cleaned_subseries: Dict[str, List[object]] = {}
        for subseries, items in subseries_mapping.items():
            if not isinstance(subseries, str):
                raise ValueError(f"Subseries labels under '{series}' must be strings.")
            if not isinstance(items, list):
                raise ValueError(
                    f"Subseries '{subseries}' under '{series}' must map to a list."
                )
            cleaned_subseries[subseries] = items
        result[series] = cleaned_subseries
    return result


def collapse_subseries_mapping(
    *,
    series_mapping: SeriesMapping,
    series_to_subseries: SeriesToSubseriesMapping,
) -> SeriesMapping:
    """Collapse flat subseries labels back to original series labels."""
    label_to_original = _build_label_to_original(series_to_subseries)
    unknown_labels = [label for label in series_mapping if label not in label_to_original]
    if unknown_labels:
        preview = ", ".join(unknown_labels[:20])
        more = "" if len(unknown_labels) <= 20 else f", ... and {len(unknown_labels) - 20} more"
        raise ValueError(
            "The following series-json labels were not found in "
            f"series-to-subseries-json: {preview}{more}"
        )

    collapsed: SeriesMapping = {}
    for original_series, subseries_mapping in series_to_subseries.items():
        values: List[object] = []
        for label in _candidate_labels(original_series, subseries_mapping):
            if label in series_mapping:
                values.extend(series_mapping[label])
        if values:
            collapsed[original_series] = _dedupe_preserving_order(values)
    return collapsed


def write_collapsed_series_mapping(
    *,
    series_json: Path,
    series_to_subseries_json: Path,
    output_json: Path,
) -> Path:
    """Load, collapse, and write a series mapping."""
    series_mapping = load_series_mapping(series_json)
    series_to_subseries = load_series_to_subseries_mapping(series_to_subseries_json)
    collapsed = collapse_subseries_mapping(
        series_mapping=series_mapping,
        series_to_subseries=series_to_subseries,
    )
    save_json(collapsed, output_json)
    _log_summary(
        input_series_count=len(series_mapping),
        output_series_count=len(collapsed),
        output_item_count=sum(len(items) for items in collapsed.values()),
        output_json=output_json,
    )
    return output_json


def _build_label_to_original(
    series_to_subseries: SeriesToSubseriesMapping,
) -> Dict[str, str]:
    label_to_original: Dict[str, str] = {}
    for original_series, subseries_mapping in series_to_subseries.items():
        _add_label_mapping(label_to_original, original_series, original_series)
        for subseries_label in subseries_mapping:
            _add_label_mapping(label_to_original, subseries_label, original_series)
    return label_to_original


def _add_label_mapping(
    label_to_original: Dict[str, str], label: str, original_series: str
) -> None:
    existing = label_to_original.get(label)
    if existing is not None and existing != original_series:
        raise ValueError(
            f"Label '{label}' maps to both '{existing}' and '{original_series}'."
        )
    label_to_original[label] = original_series


def _candidate_labels(
    original_series: str, subseries_mapping: Dict[str, List[object]]
) -> List[str]:
    labels = [original_series]
    labels.extend(label for label in subseries_mapping if label != original_series)
    return labels


def _dedupe_preserving_order(values: List[object]) -> List[object]:
    seen: set[str] = set()
    deduped: List[object] = []
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _log_summary(
    *,
    input_series_count: int,
    output_series_count: int,
    output_item_count: int,
    output_json: Path,
) -> None:
    log("Collapse summary:")
    log(f"  input: {input_series_count} series/subseries labels")
    log(
        "  output: "
        f"{output_series_count} series, {plural(output_item_count, 'item')}"
    )
    log(f"  output_json: {output_json}")


def main() -> None:
    """Entry point for the standalone collapse script."""
    args = parse_args()
    write_collapsed_series_mapping(
        series_json=Path(args.series_json).resolve(),
        series_to_subseries_json=Path(args.series_to_subseries_json).resolve(),
        output_json=Path(args.output_json).resolve(),
    )


if __name__ == "__main__":
    main()
