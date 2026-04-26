import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from clip_image_similarity.clustering import (
    build_flat_subseries_mapping,
    build_precomputed_distance_matrix,
    cluster_single_series,
)
from clip_image_similarity.packed_distances import PackedDistances
from metrics.series_indices import validate_series_indices


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "cluster_series_by_distance.py"
)
SPEC = importlib.util.spec_from_file_location("cluster_series_by_distance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_series_to_image_paths = MODULE.build_series_to_image_paths
generate_clustered_series_labels = MODULE.generate_clustered_series_labels
parse_args = MODULE.parse_args


def _flatten_upper_triangle_numpy(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(float(matrix[i, j]))
    return np.asarray(values, dtype=np.float32)


def _write_run_output(
    tmp_path: Path,
    matrix: np.ndarray,
    series_to_indices: dict[str, list[int]],
    image_paths: list[str] | None = None,
) -> Path:
    out_dir = tmp_path / "run_output"
    eval_dir = out_dir / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        eval_dir / "pairwise_distances.npz",
        distances=_flatten_upper_triangle_numpy(matrix),
        dtype="float32",
        allow_pickle=False,
    )
    if image_paths is None:
        image_paths = [f"/dataset/img_{idx}.png" for idx in range(matrix.shape[0])]
    (out_dir / "image_paths.json").write_text(json.dumps(image_paths), encoding="utf-8")
    (out_dir / "series_to_indices.json").write_text(
        json.dumps(series_to_indices), encoding="utf-8"
    )
    return out_dir


def _require_sklearn():
    return pytest.importorskip("sklearn")


def test_build_precomputed_distance_matrix_uses_requested_indices():
    matrix = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    distances = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    result = build_precomputed_distance_matrix(distances, [3, 1, 0])

    expected = np.array(
        [
            [0.0, 0.5, 0.3],
            [0.5, 0.0, 0.1],
            [0.3, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected)


def test_average_linkage_can_merge_through_cluster_average():
    _require_sklearn()
    matrix = np.array(
        [
            [0.0, 0.1, 0.7],
            [0.1, 0.0, 0.2],
            [0.7, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    distances = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    result = cluster_single_series(
        series_name="seriesA",
        indices=[0, 1, 2],
        distances=distances,
        distance_threshold=0.5,
        linkage="average",
        min_samples=2,
    )

    assert build_flat_subseries_mapping([result]) == {"seriesA": [0, 1, 2]}


def test_complete_linkage_keeps_far_pair_separate():
    _require_sklearn()
    matrix = np.array(
        [
            [0.0, 0.1, 0.7],
            [0.1, 0.0, 0.2],
            [0.7, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    distances = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    result = cluster_single_series(
        series_name="seriesA",
        indices=[0, 1, 2],
        distances=distances,
        distance_threshold=0.5,
        linkage="complete",
        min_samples=1,
    )

    assert build_flat_subseries_mapping([result]) == {
        "seriesA_subseries_0": [0, 1],
        "seriesA_subseries_1": [2],
    }


def test_min_samples_drops_small_subseries():
    _require_sklearn()
    matrix = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.9],
            [0.9, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    distances = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    result = cluster_single_series(
        series_name="seriesA",
        indices=[0, 1, 2],
        distances=distances,
        distance_threshold=0.2,
        linkage="average",
        min_samples=2,
    )

    assert build_flat_subseries_mapping([result]) == {"seriesA": [0, 1]}
    assert result.dropped_subseries_count == 1
    assert result.dropped_image_count == 1


def test_generate_clustered_series_labels_clusters_per_original_series(tmp_path: Path):
    _require_sklearn()
    matrix = np.array(
        [
            [0.0, 0.1, 0.05, 0.9],
            [0.1, 0.0, 0.05, 0.9],
            [0.05, 0.05, 0.0, 0.1],
            [0.9, 0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1], "B": [2, 3]})

    run_dir = generate_clustered_series_labels(
        pairwise_output_dir=out_dir,
        distance_threshold=0.2,
        linkage="average",
        min_samples=2,
    )

    subseries = json.loads((run_dir / "series_to_indices.json").read_text())
    nested = json.loads((run_dir / "series_to_subseries_indices.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())

    assert run_dir.name == "clustered_series_by_distance_t0_2_average_min2"
    assert subseries == {"A": [0, 1], "B": [2, 3]}
    assert nested == {
        "A": {"A": [0, 1]},
        "B": {"B": [2, 3]},
    }
    assert validate_series_indices(subseries, n_items=4) == subseries
    assert summary["retained_subseries_count"] == 2
    assert summary["retained_image_count"] == 4
    assert summary["series"]["A"]["retained_subseries_count"] == 1
    assert summary["series"]["A"]["subseries"]["A"]["image_count"] == 2


def test_generate_clustered_series_labels_writes_image_paths(tmp_path: Path):
    _require_sklearn()
    matrix = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1]})

    run_dir = generate_clustered_series_labels(
        pairwise_output_dir=out_dir,
        distance_threshold=0.2,
        min_samples=2,
    )

    paths = json.loads((run_dir / "series_to_image_paths.json").read_text())
    assert paths == {"A": ["/dataset/img_0.png", "/dataset/img_1.png"]}


@pytest.mark.parametrize(
    ("missing_relpath", "expected_message"),
    [
        ("evaluation_results/pairwise_distances.npz", "Distances file not found"),
        ("series_to_indices.json", "Series indices file not found"),
        ("image_paths.json", "Image paths file not found"),
    ],
)
def test_generate_clustered_series_labels_missing_inputs(
    tmp_path: Path, missing_relpath: str, expected_message: str
):
    matrix = np.eye(2, dtype=np.float32)
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1]})
    (out_dir / missing_relpath).unlink()

    with pytest.raises(FileNotFoundError, match=expected_message):
        generate_clustered_series_labels(
            pairwise_output_dir=out_dir,
            distance_threshold=0.5,
        )


def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "script",
            "--pairwise-output-dir",
            "/tmp/output",
            "--distance-threshold",
            "0.25",
            "--linkage",
            "complete",
            "--min-samples",
            "3",
        ],
    )

    args = parse_args()

    assert args.pairwise_output_dir == "/tmp/output"
    assert args.distance_threshold == pytest.approx(0.25)
    assert args.linkage == "complete"
    assert args.min_samples == 3


def test_main_entry_point(tmp_path: Path):
    _require_sklearn()
    matrix = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1]})

    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            "script",
            "--pairwise-output-dir",
            str(out_dir),
            "--distance-threshold",
            "0.20",
            "--min-samples",
            "2",
        ]
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.argv = original_argv

    created = sorted(out_dir.glob("clustered_series_by_distance_t0_2_average_min2"))
    assert len(created) == 1
    assert json.loads((created[0] / "series_to_indices.json").read_text()) == {
        "A": [0, 1]
    }
