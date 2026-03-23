import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clip_image_similarity.packed_distances import PackedDistances
from metrics.similarity_samples import (
    PairSampleSpec,
    build_percentile_bins_from_values,
    load_image_paths,
    write_similarity_samples,
)


def _make_symmetric_distance_matrix(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(i * 100 + j)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _flatten_upper_triangle_numpy(matrix: np.ndarray) -> np.ndarray:
    values = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            values.append(float(matrix[i, j]))
    return np.asarray(values, dtype=np.float32)


def _write_image_paths_json(out_dir: Path, count: int) -> None:
    image_paths = []
    for idx in range(count):
        image_path = out_dir / f"sample_{idx}.png"
        Image.new("RGB", (16, 12), color=(idx * 30, idx * 20, idx * 10)).save(image_path)
        image_paths.append(image_path.as_posix())
    (out_dir / "image_paths.json").write_text(json.dumps(image_paths), encoding="utf-8")


def test_build_percentile_bins_from_values_orders_by_distance():
    bins = build_percentile_bins_from_values(
        np.asarray([0.9, 0.1, 0.3, 0.7], dtype=np.float32), num_percentile_bins=2
    )

    assert [bin_info.name for bin_info in bins] == ["pct_00_50", "pct_50_100"]
    assert [bin_info.record_count for bin_info in bins] == [2, 2]
    assert bins[0].distance_min == pytest.approx(0.1)
    assert bins[0].distance_max == pytest.approx(0.3)
    assert bins[1].distance_min == pytest.approx(0.7)
    assert bins[1].distance_max == pytest.approx(0.9)


def test_load_image_paths_validates_expected_count(tmp_path: Path):
    _write_image_paths_json(tmp_path, count=2)

    loaded = load_image_paths(tmp_path, expected_count=2)

    assert loaded == [tmp_path / "sample_0.png", tmp_path / "sample_1.png"]


def test_write_similarity_samples_creates_manifests_and_examples(tmp_path: Path):
    n = 4
    matrix = _make_symmetric_distance_matrix(n)
    flat = _flatten_upper_triangle_numpy(matrix)
    out_dir = tmp_path / "run_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_image_paths_json(out_dir, count=n)

    distances = PackedDistances(flat)
    series_to_indices = {"A": [0, 1], "B": [2, 3]}
    sample_dir = write_similarity_samples(
        pairwise_output_dir=out_dir,
        distances=distances,
        series_to_indices=series_to_indices,
        labeled_images_only=True,
        spec=PairSampleSpec(samples_per_bin=1, num_percentile_bins=2),
        within_values=np.asarray([matrix[0, 1], matrix[2, 3]], dtype=np.float32),
        out_values=np.asarray(
            [
                matrix[0, 2],
                matrix[0, 3],
                matrix[1, 2],
                matrix[1, 3],
                matrix[2, 0],
                matrix[2, 1],
                matrix[3, 0],
                matrix[3, 1],
            ],
            dtype=np.float32,
        ),
    )

    within_manifest = json.loads((sample_dir / "within_series" / "manifest.json").read_text())
    out_manifest = json.loads((sample_dir / "out_of_series" / "manifest.json").read_text())

    assert within_manifest["record_count"] == 2
    assert within_manifest["bin_count"] == 2
    assert out_manifest["record_count"] == 8
    assert out_manifest["bin_count"] == 2
    assert within_manifest["bins"][0]["samples"][0]["similarity"] == 1.0 - matrix[0, 1]

    for manifest in (within_manifest, out_manifest):
        for bin_info in manifest["bins"]:
            assert len(bin_info["samples"]) == 1
            sample_file = sample_dir / bin_info["samples"][0]["file"]
            assert sample_file.is_file()
