from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from clip_image_similarity.image_loader import load_rgb_image
from clip_image_similarity.packed_distances import PackedDistances
from clip_image_similarity.serialization import save_json
from clip_image_similarity.utils import log, plural


@dataclass(frozen=True)
class PairSampleSpec:
    """Configuration for saving representative similarity samples."""

    samples_per_bin: int
    num_percentile_bins: int
    panel_size_px: int = 384
    render_dpi: int = 150
    sample_buffer_multiplier: int = 50
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.samples_per_bin <= 0:
            raise ValueError("samples_per_bin must be positive.")
        if self.num_percentile_bins <= 0:
            raise ValueError("num_percentile_bins must be positive.")
        if self.panel_size_px <= 0:
            raise ValueError("panel_size_px must be positive.")
        if self.render_dpi <= 0:
            raise ValueError("render_dpi must be positive.")
        if self.sample_buffer_multiplier <= 0:
            raise ValueError("sample_buffer_multiplier must be positive.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")


@dataclass(frozen=True)
class PairRecord:
    """Metadata for one pairwise similarity example."""

    pool: str
    left_index: int
    right_index: int
    distance: float
    similarity: float
    left_series_label: str
    right_series_label: str


@dataclass(frozen=True)
class PercentileBin:
    """Metadata for one percentile bin within a pool."""

    bin_index: int
    name: str
    percentile_start: int
    percentile_end: int
    lower_bound: float
    upper_bound: float
    record_count: int
    distance_min: float | None
    distance_max: float | None


def load_image_paths(pairwise_output_dir: Path, expected_count: int) -> List[Path]:
    """Load the matrix-aligned image path ordering for a pairwise run."""

    image_paths_path = pairwise_output_dir / "image_paths.json"
    if not image_paths_path.is_file():
        raise FileNotFoundError(f"Image paths file not found: {image_paths_path}")

    with image_paths_path.open("r", encoding="utf-8") as f:
        raw_paths = json.load(f)
    if not isinstance(raw_paths, list) or any(not isinstance(p, str) for p in raw_paths):
        raise ValueError("image_paths.json must contain a JSON list of file path strings.")
    if len(raw_paths) != expected_count:
        raise ValueError(
            "image_paths.json length does not match pairwise distances size: "
            f"expected {expected_count}, found {len(raw_paths)}."
        )
    return [Path(p) for p in raw_paths]


def build_percentile_bins_from_values(
    values: np.ndarray, num_percentile_bins: int
) -> List[PercentileBin]:
    """Build percentile bins directly from a pool of distances."""

    if num_percentile_bins <= 0:
        raise ValueError("num_percentile_bins must be positive.")
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return []

    quantile_inputs = values.astype(np.float64, copy=False)
    quantiles = np.quantile(
        quantile_inputs,
        q=np.linspace(0.0, 1.0, num_percentile_bins + 1, dtype=np.float64),
    )
    assignments = _assign_values_to_bin_indices(values, quantiles)
    bins: List[PercentileBin] = []
    for bin_idx in range(num_percentile_bins):
        bin_values = values[assignments == bin_idx]
        pct_start = int(round(100 * bin_idx / num_percentile_bins))
        pct_end = int(round(100 * (bin_idx + 1) / num_percentile_bins))
        bins.append(
            PercentileBin(
                bin_index=bin_idx,
                name=f"pct_{pct_start:02d}_{pct_end:02d}",
                percentile_start=pct_start,
                percentile_end=pct_end,
                lower_bound=float(quantiles[bin_idx]),
                upper_bound=float(quantiles[bin_idx + 1]),
                record_count=int(bin_values.size),
                distance_min=float(bin_values.min()) if bin_values.size else None,
                distance_max=float(bin_values.max()) if bin_values.size else None,
            )
        )
    return bins


def write_similarity_samples(
    *,
    pairwise_output_dir: Path,
    distances: PackedDistances,
    series_to_indices: Dict[str, List[int]],
    labeled_images_only: bool,
    spec: PairSampleSpec,
    within_values: np.ndarray,
    out_values: np.ndarray,
) -> Path:
    """Save representative within/out-of-series image pairs grouped by percentile."""

    pairwise_output_dir = pairwise_output_dir.resolve()
    image_paths = load_image_paths(pairwise_output_dir, distances.n)
    output_dir = _make_samples_output_dir(pairwise_output_dir)
    pool_bins = {
        "within_series": build_percentile_bins_from_values(
            within_values, spec.num_percentile_bins
        ),
        "out_of_series": build_percentile_bins_from_values(
            out_values, spec.num_percentile_bins
        ),
    }
    index_to_labels = _build_index_to_series_labels(series_to_indices)
    labeled_union_idxs = _labeled_union(series_to_indices)
    buffer_size = spec.samples_per_bin * spec.sample_buffer_multiplier
    collected_by_pool = {
        "within_series": _collect_within_samples(
            distances=distances,
            series_to_indices=series_to_indices,
            bins=pool_bins["within_series"],
            buffer_size=buffer_size,
        ),
        "out_of_series": _collect_out_samples(
            distances=distances,
            series_to_indices=series_to_indices,
            labeled_images_only=labeled_images_only,
            labeled_union_idxs=labeled_union_idxs,
            index_to_labels=index_to_labels,
            bins=pool_bins["out_of_series"],
            buffer_size=buffer_size,
        ),
    }
    rng = np.random.default_rng(spec.random_seed)
    selected_by_pool = {
        pool_name: {
            pct_bin.bin_index: _sample_records(
                collected_by_pool[pool_name][pct_bin.bin_index],
                samples_per_bin=spec.samples_per_bin,
                rng=rng,
            )
            for pct_bin in bins
            if pct_bin.record_count > 0
        }
        for pool_name, bins in pool_bins.items()
    }
    total_samples = sum(
        len(records)
        for selected_bins in selected_by_pool.values()
        for records in selected_bins.values()
    )

    with tqdm(total=total_samples, desc="Saving similarity samples") as progress:
        for pool_name, bins in pool_bins.items():
            pool_dir = output_dir / pool_name
            pool_dir.mkdir(parents=True, exist_ok=True)
            manifest_bins = []
            for pct_bin in bins:
                if pct_bin.record_count == 0:
                    continue
                selected = selected_by_pool[pool_name][pct_bin.bin_index]
                bin_dir = pool_dir / pct_bin.name
                bin_dir.mkdir(parents=True, exist_ok=True)
                manifest_samples = []
                for sample_idx, record in enumerate(selected, start=1):
                    filename = f"example_{sample_idx:03d}.png"
                    output_path = bin_dir / filename
                    _render_pair_sample(
                        output_path=output_path,
                        record=record,
                        image_paths=image_paths,
                        spec=spec,
                    )
                    progress.update(1)
                    manifest_samples.append(
                        {
                            "file": str(Path(pool_name) / pct_bin.name / filename),
                            "left_index": record.left_index,
                            "right_index": record.right_index,
                            "distance": record.distance,
                            "similarity": record.similarity,
                            "left_series_label": record.left_series_label,
                            "right_series_label": record.right_series_label,
                            "left_image_path": image_paths[record.left_index].as_posix(),
                            "right_image_path": image_paths[record.right_index].as_posix(),
                        }
                    )

                manifest_bins.append(
                    {
                        "name": pct_bin.name,
                        "percentile_start": pct_bin.percentile_start,
                        "percentile_end": pct_bin.percentile_end,
                        "lower_bound": pct_bin.lower_bound,
                        "upper_bound": pct_bin.upper_bound,
                        "record_count": pct_bin.record_count,
                        "distance_min": pct_bin.distance_min,
                        "distance_max": pct_bin.distance_max,
                        "samples": manifest_samples,
                    }
                )

            save_json(
                {
                    "pool": pool_name,
                    "samples_per_bin": spec.samples_per_bin,
                    "num_percentile_bins": spec.num_percentile_bins,
                    "sample_buffer_multiplier": spec.sample_buffer_multiplier,
                    "bin_count": len(manifest_bins),
                    "record_count": int(sum(pct_bin.record_count for pct_bin in bins)),
                    "bins": manifest_bins,
                },
                pool_dir / "manifest.json",
            )
            log(
                f"Wrote {plural(sum(len(bin_info['samples']) for bin_info in manifest_bins), 'sample')} "
                f"across {plural(len(manifest_bins), 'percentile bin')} for pool '{pool_name}'."
            )

    log(f"Saved similarity samples under {output_dir}")
    return output_dir


def _build_index_to_series_labels(series_to_indices: Dict[str, List[int]]) -> Dict[int, str]:
    index_to_names: Dict[int, List[str]] = {}
    for series_name, indices in series_to_indices.items():
        for index in sorted(set(int(x) for x in indices)):
            index_to_names.setdefault(index, []).append(series_name)
    return {
        index: ",".join(sorted(names))
        for index, names in index_to_names.items()
    }


def _labeled_union(series_to_indices: Dict[str, List[int]]) -> List[int]:
    labeled: set[int] = set()
    for idxs in series_to_indices.values():
        labeled.update(int(x) for x in idxs)
    return sorted(labeled)


def _series_out_candidates(
    *,
    n_items: int,
    series_set: set[int],
    labeled_union_idxs: List[int],
    labeled_images_only: bool,
) -> List[int]:
    if labeled_images_only:
        return [i for i in labeled_union_idxs if i not in series_set]
    return [i for i in range(n_items) if i not in series_set]


def _make_samples_output_dir(pairwise_output_dir: Path) -> Path:
    root = pairwise_output_dir / "similarity_samples"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _collect_within_samples(
    *,
    distances: PackedDistances,
    series_to_indices: Dict[str, List[int]],
    bins: Sequence[PercentileBin],
    buffer_size: int,
) -> Dict[int, List[PairRecord]]:
    collected, target_counts = _initialize_collectors(bins, buffer_size)
    if not target_counts:
        return collected

    boundaries = _bin_boundaries(bins)
    remaining = len(target_counts)
    for series_name, idxs in series_to_indices.items():
        series = sorted(set(int(x) for x in idxs))
        for left_pos in range(len(series)):
            for right_pos in range(left_pos + 1, len(series)):
                i = series[left_pos]
                j = series[right_pos]
                distance = distances.distance(i, j)
                bin_index = _assign_distance_to_bin_index(distance, boundaries)
                if bin_index not in target_counts:
                    continue
                bucket = collected[bin_index]
                if len(bucket) >= target_counts[bin_index]:
                    continue
                bucket.append(
                    PairRecord(
                        pool="within_series",
                        left_index=i,
                        right_index=j,
                        distance=distance,
                        similarity=1.0 - distance,
                        left_series_label=series_name,
                        right_series_label=series_name,
                    )
                )
                if len(bucket) == target_counts[bin_index]:
                    remaining -= 1
                    if remaining == 0:
                        return collected
    return collected


def _collect_out_samples(
    *,
    distances: PackedDistances,
    series_to_indices: Dict[str, List[int]],
    labeled_images_only: bool,
    labeled_union_idxs: List[int],
    index_to_labels: Dict[int, str],
    bins: Sequence[PercentileBin],
    buffer_size: int,
) -> Dict[int, List[PairRecord]]:
    collected, target_counts = _initialize_collectors(bins, buffer_size)
    if not target_counts:
        return collected

    boundaries = _bin_boundaries(bins)
    remaining = len(target_counts)
    for series_name, idxs in series_to_indices.items():
        series = sorted(set(int(x) for x in idxs))
        series_set = set(series)
        out_candidates = _series_out_candidates(
            n_items=distances.n,
            series_set=series_set,
            labeled_union_idxs=labeled_union_idxs,
            labeled_images_only=labeled_images_only,
        )
        for i in series:
            for j in out_candidates:
                distance = distances.distance(i, j)
                bin_index = _assign_distance_to_bin_index(distance, boundaries)
                if bin_index not in target_counts:
                    continue
                bucket = collected[bin_index]
                if len(bucket) >= target_counts[bin_index]:
                    continue
                bucket.append(
                    PairRecord(
                        pool="out_of_series",
                        left_index=i,
                        right_index=int(j),
                        distance=distance,
                        similarity=1.0 - distance,
                        left_series_label=series_name,
                        right_series_label=index_to_labels.get(int(j), "unlabeled"),
                    )
                )
                if len(bucket) == target_counts[bin_index]:
                    remaining -= 1
                    if remaining == 0:
                        return collected
    return collected


def _initialize_collectors(
    bins: Sequence[PercentileBin], buffer_size: int
) -> tuple[Dict[int, List[PairRecord]], Dict[int, int]]:
    collected = {pct_bin.bin_index: [] for pct_bin in bins if pct_bin.record_count > 0}
    target_counts = {
        pct_bin.bin_index: min(buffer_size, pct_bin.record_count)
        for pct_bin in bins
        if pct_bin.record_count > 0
    }
    return collected, target_counts


def _bin_boundaries(bins: Sequence[PercentileBin]) -> np.ndarray:
    if not bins:
        return np.asarray([], dtype=np.float64)
    return np.asarray([pct_bin.upper_bound for pct_bin in bins[:-1]], dtype=np.float64)


def _assign_values_to_bin_indices(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    if quantiles.size < 2:
        raise ValueError("quantiles must contain at least two entries.")
    if quantiles.size == 2:
        return np.zeros(values.shape, dtype=np.int64)
    return np.searchsorted(quantiles[1:-1], values, side="right")


def _assign_distance_to_bin_index(distance: float, boundaries: np.ndarray) -> int:
    if boundaries.size == 0:
        return 0
    return int(np.searchsorted(boundaries, distance, side="right"))


def _sample_records(
    records: Sequence[PairRecord], samples_per_bin: int, rng: np.random.Generator
) -> List[PairRecord]:
    ordered = list(records)
    if len(ordered) <= samples_per_bin:
        return ordered
    selected_positions = np.sort(rng.choice(len(ordered), size=samples_per_bin, replace=False))
    return [ordered[int(position)] for position in selected_positions]


def _render_pair_sample(
    *,
    output_path: Path,
    record: PairRecord,
    image_paths: Sequence[Path],
    spec: PairSampleSpec,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    left_image = _prepare_panel_image(image_paths[record.left_index], spec.panel_size_px)
    right_image = _prepare_panel_image(image_paths[record.right_index], spec.panel_size_px)

    pad = 16
    panel_w = spec.panel_size_px
    panel_h = spec.panel_size_px
    header_lines = [
        f"pool: {record.pool.replace('_', ' ')}",
        f"distance: {record.distance:.6f} | similarity: {record.similarity:.6f}",
        f"indices: ({record.left_index}, {record.right_index})",
    ]
    left_lines = [
        f"left series: {record.left_series_label}",
        f"file: {_shorten_path(image_paths[record.left_index])}",
    ]
    right_lines = [
        f"right series: {record.right_series_label}",
        f"file: {_shorten_path(image_paths[record.right_index])}",
    ]

    canvas_width = panel_w * 2 + pad * 3
    header_height = _text_block_height(header_lines, font) + pad * 2
    footer_height = max(
        _text_block_height(left_lines, font),
        _text_block_height(right_lines, font),
    ) + pad * 2
    canvas_height = header_height + panel_h + footer_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((pad, pad), "\n".join(header_lines), fill="black", font=font, spacing=4)

    left_box = (pad, header_height, pad + panel_w, header_height + panel_h)
    right_box = (
        pad * 2 + panel_w,
        header_height,
        pad * 2 + panel_w * 2,
        header_height + panel_h,
    )
    _paste_centered(canvas, left_image, left_box)
    _paste_centered(canvas, right_image, right_box)

    draw.rectangle(left_box, outline="black", width=1)
    draw.rectangle(right_box, outline="black", width=1)

    footer_y = header_height + panel_h + pad
    draw.multiline_text((pad, footer_y), "\n".join(left_lines), fill="black", font=font, spacing=4)
    draw.multiline_text(
        (pad * 2 + panel_w, footer_y),
        "\n".join(right_lines),
        fill="black",
        font=font,
        spacing=4,
    )
    canvas.save(output_path, dpi=(spec.render_dpi, spec.render_dpi))


def _prepare_panel_image(path: Path, panel_size_px: int) -> Image.Image:
    rgb = load_rgb_image(path)
    return ImageOps.contain(
        rgb,
        (panel_size_px - 8, panel_size_px - 8),
        method=Image.Resampling.LANCZOS,
    )


def _paste_centered(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    left = box[0] + (box_width - image.width) // 2
    top = box[1] + (box_height - image.height) // 2
    canvas.paste(image, (left, top))


def _text_block_height(lines: Sequence[str], font: ImageFont.ImageFont) -> int:
    temp = Image.new("RGB", (1, 1), color="white")
    draw = ImageDraw.Draw(temp)
    _, _, _, bottom = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=4)
    return bottom


def _shorten_path(path: Path, width: int = 44) -> str:
    return textwrap.shorten(path.name or path.as_posix(), width=width, placeholder="...")
