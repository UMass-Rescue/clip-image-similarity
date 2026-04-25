import json
from pathlib import Path

import numpy as np

import pytest

from clip_image_similarity.packed_distances import PackedDistances
from metrics.distance_histogram import DistanceHistogramRunner, HistogramSpec, SeriesDistanceExtractor


def _make_symmetric_distance_matrix(n: int) -> np.ndarray:
    """Create a deterministic symmetric distance matrix with unique off-diagonal values."""
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(i * 100 + j)
            m[i, j] = v
            m[j, i] = v
    return m


def _flatten_upper_triangle_numpy(m: np.ndarray) -> np.ndarray:
    """Flatten upper-triangular values (i<j) in row-major order."""
    n = m.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(m[i, j]))
    return np.asarray(out, dtype=np.float32)


def test_series_distance_extractor_within_and_out_basic():
    m = _make_symmetric_distance_matrix(6)
    packed = PackedDistances(_flatten_upper_triangle_numpy(m))
    extractor = SeriesDistanceExtractor(packed)

    series = [0, 1]
    within = extractor.within_series_distances(series)
    assert within.tolist() == [m[0, 1]]

    # Out-of-series candidates exclude series indices.
    candidates = np.asarray([2, 3, 4, 5], dtype=np.int64)
    out = extractor.out_of_series_distances(series, candidates)
    expected = [m[0, 2], m[0, 3], m[0, 4], m[0, 5], m[1, 2], m[1, 3], m[1, 4], m[1, 5]]
    assert sorted(out.tolist()) == sorted(expected)


def test_runner_creates_expected_outputs(tmp_path: Path):
    pytest.importorskip("matplotlib")

    n = 6
    m = _make_symmetric_distance_matrix(n)
    flat = _flatten_upper_triangle_numpy(m)

    out_dir = tmp_path / "run_output"
    eval_dir = out_dir / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(eval_dir / "pairwise_distances.npz", distances=flat, dtype="float32", allow_pickle=False)

    # Two labeled series; indices 4 and 5 are unlabeled.
    series_to_indices = {"A": [0, 1], "B": [2, 3]}
    (out_dir / "series_to_indices.json").write_text(json.dumps(series_to_indices), encoding="utf-8")

    spec = HistogramSpec(bins=10, range_min=0.0, range_max=600.0, density=False)
    runner = DistanceHistogramRunner(spec=spec)
    run_dir = runner.run(
        pairwise_output_dir=out_dir,
        labeled_images_only=True,
        save_raw_values=True,
    )

    histogram_dir = run_dir / "histogram"
    raw_dir = run_dir / "raw_values"
    assert (histogram_dir / "A.png").is_file()
    assert (histogram_dir / "B.png").is_file()
    assert (histogram_dir / "all_series.png").is_file()

    a_raw = np.load(raw_dir / "A.npz", allow_pickle=False)
    assert set(a_raw.files) == {"within", "out"}
    assert a_raw["within"].tolist() == [m[0, 1]]
    # labeled_images_only=True => for series A, out candidates are {2,3}
    assert sorted(a_raw["out"].tolist()) == sorted([m[0, 2], m[0, 3], m[1, 2], m[1, 3]])

    # Overall raw values are intentionally not written; they can be reconstructed
    # by combining the per-series raw files.
    assert not (raw_dir / "all_series.npz").exists()


