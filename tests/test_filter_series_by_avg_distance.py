import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from clip_image_similarity.packed_distances import PackedDistances


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "filter_series_by_avg_distance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "filter_series_by_avg_distance", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FilterStats = MODULE.FilterStats
build_series_to_image_paths = MODULE.build_series_to_image_paths
filter_series_to_indices = MODULE.filter_series_to_indices
generate_filtered_series_labels = MODULE.generate_filtered_series_labels
parse_args = MODULE.parse_args


def _flatten_upper_triangle_numpy(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(float(m[i, j]))
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


def test_filter_series_to_indices_removes_image_failing_in_series_only():
    matrix = np.array(
        [
            [0.0, 0.1, 0.9, 1.0],
            [0.1, 0.0, 0.9, 1.0],
            [0.9, 0.9, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    dist = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    filtered, stats = filter_series_to_indices(
        {"A": [0, 1, 2]},
        dist,
        in_series_threshold=0.5,
        out_of_series_threshold=0.5,
    )

    assert filtered == {"A": [0, 1]}
    assert stats.removed_by_in_series_threshold_count == 1
    assert stats.removed_by_out_of_series_threshold_count == 0


def test_filter_series_to_indices_removes_image_failing_out_of_series_only():
    matrix = np.array(
        [
            [0.0, 0.1, 0.1, 1.0, 1.0],
            [0.1, 0.0, 0.1, 1.0, 1.0],
            [0.1, 0.1, 0.0, 0.1, 0.1],
            [1.0, 1.0, 0.1, 0.0, 0.2],
            [1.0, 1.0, 0.1, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    dist = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    filtered, stats = filter_series_to_indices(
        {"A": [0, 1, 2]},
        dist,
        in_series_threshold=0.2,
        out_of_series_threshold=0.5,
    )

    assert filtered == {"A": [0, 1]}
    assert stats.removed_by_in_series_threshold_count == 0
    assert stats.removed_by_out_of_series_threshold_count == 1


def test_filter_series_to_indices_keeps_boundary_values():
    matrix = np.array(
        [
            [0.0, 0.25, 0.6, 0.6],
            [0.25, 0.0, 0.6, 0.6],
            [0.6, 0.6, 0.0, 0.2],
            [0.6, 0.6, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    dist = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    filtered, _ = filter_series_to_indices(
        {"A": [0, 1]},
        dist,
        in_series_threshold=0.25,
        out_of_series_threshold=0.6,
    )

    assert filtered == {"A": [0, 1]}


def test_filter_series_to_indices_counts_images_failing_both_thresholds_once():
    matrix = np.array(
        [
            [0.0, 0.1, 0.9, 1.0, 1.0],
            [0.1, 0.0, 0.9, 1.0, 1.0],
            [0.9, 0.9, 0.0, 0.1, 0.1],
            [1.0, 1.0, 0.1, 0.0, 0.2],
            [1.0, 1.0, 0.1, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    dist = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    filtered, stats = filter_series_to_indices(
        {"A": [0, 1, 2]},
        dist,
        in_series_threshold=0.5,
        out_of_series_threshold=0.5,
    )

    assert filtered == {"A": [0, 1]}
    assert stats.removed_by_in_series_threshold_count == 1
    assert stats.removed_by_out_of_series_threshold_count == 1
    assert stats.removed_by_both_thresholds_count == 1
    assert stats.unique_threshold_failed_image_count == 1


def test_filter_series_to_indices_uses_original_series_membership_single_pass():
    matrix = np.array(
        [
            [0.0, 0.9, 0.9, 1.0],
            [0.9, 0.0, 0.1, 0.4],
            [0.9, 0.1, 0.0, 0.4],
            [1.0, 0.4, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    dist = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    filtered, stats = filter_series_to_indices(
        {"A": [0, 1, 2]},
        dist,
        in_series_threshold=0.5,
        out_of_series_threshold=0.6,
    )

    assert filtered == {}
    assert stats.retained_image_count == 0


def test_filter_series_to_indices_uses_all_non_series_images_for_out_average():
    matrix = np.array(
        [
            [0.0, 0.1, 1.0, 1.0, 0.1],
            [0.1, 0.0, 1.0, 1.0, 0.1],
            [1.0, 1.0, 0.0, 0.2, 0.9],
            [1.0, 1.0, 0.2, 0.0, 0.9],
            [0.1, 0.1, 0.9, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    dist = PackedDistances(_flatten_upper_triangle_numpy(matrix))

    filtered, _ = filter_series_to_indices(
        {"A": [0, 1], "B": [2, 3]},
        dist,
        in_series_threshold=0.2,
        out_of_series_threshold=0.8,
    )

    assert filtered == {"B": [2, 3]}


def test_build_series_to_image_paths_uses_image_order():
    image_paths = [Path("/dataset/a.png"), Path("/dataset/b.png"), Path("/dataset/c.png")]

    result = build_series_to_image_paths({"series": [2, 0]}, image_paths)

    assert result == {"series": ["/dataset/c.png", "/dataset/a.png"]}


def test_generate_filtered_series_labels_writes_expected_outputs_and_summary(
    tmp_path: Path,
    capsys,
):
    matrix = np.array(
        [
            [0.0, 0.1, 0.9, 1.0, 1.0, 0.4],
            [0.1, 0.0, 0.9, 1.0, 1.0, 0.4],
            [0.9, 0.9, 0.0, 0.1, 0.1, 0.4],
            [1.0, 1.0, 0.1, 0.0, 0.2, 1.0],
            [1.0, 1.0, 0.1, 0.2, 0.0, 1.0],
            [0.4, 0.4, 0.4, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    out_dir = _write_run_output(
        tmp_path,
        matrix,
        {"A": [0, 1, 2], "B": [3, 4], "C": [5]},
    )

    run_dir = generate_filtered_series_labels(
        out_dir,
        in_series_threshold=0.5,
        out_of_series_threshold=0.5,
        in_series_threshold_label="0.50",
        out_of_series_threshold_label="0.50",
    )

    assert run_dir.name == "filtered_series_by_avg_distance_in_t0_50_out_t0_50"
    filtered_indices = json.loads((run_dir / "series_to_indices.json").read_text())
    filtered_paths = json.loads((run_dir / "series_to_image_paths.json").read_text())
    assert filtered_indices == {"A": [0, 1], "B": [3, 4]}
    assert filtered_paths == {
        "A": ["/dataset/img_0.png", "/dataset/img_1.png"],
        "B": ["/dataset/img_3.png", "/dataset/img_4.png"],
    }

    captured = capsys.readouterr()
    assert "Filtering summary:" in captured.out
    assert "thresholds: in-series <= 0.5, out-of-series >= 0.5" in captured.out
    assert "input: 3 series, 6 labeled images, 6 total matrix images" in captured.out
    assert "1 image by in-series" in captured.out
    assert "1 image by out-of-series" in captured.out
    assert "1 image by both" in captured.out
    assert "1 series removed for having fewer than 2 retained images" in captured.out
    assert "output: 2 series, 4 labeled images" in captured.out


def test_generate_filtered_series_labels_drops_small_series(tmp_path: Path):
    matrix = np.array(
        [
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.9],
            [0.8, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1, 2]})

    run_dir = generate_filtered_series_labels(
        out_dir,
        in_series_threshold=0.3,
        out_of_series_threshold=0.0,
        in_series_threshold_label="0.30",
        out_of_series_threshold_label="0.00",
    )

    assert json.loads((run_dir / "series_to_indices.json").read_text()) == {}
    assert json.loads((run_dir / "series_to_image_paths.json").read_text()) == {}


@pytest.mark.parametrize(
    ("missing_relpath", "expected_message"),
    [
        ("evaluation_results/pairwise_distances.npz", "Distances file not found"),
        ("series_to_indices.json", "Series indices file not found"),
        ("image_paths.json", "Image paths file not found"),
    ],
)
def test_generate_filtered_series_labels_missing_inputs(
    tmp_path: Path, missing_relpath: str, expected_message: str
):
    matrix = np.eye(2, dtype=np.float32)
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1]})
    (out_dir / missing_relpath).unlink()

    with pytest.raises(FileNotFoundError, match=expected_message):
        generate_filtered_series_labels(
            out_dir,
            in_series_threshold=0.5,
            out_of_series_threshold=0.5,
        )


def test_generate_filtered_series_labels_rejects_out_of_bounds_indices(tmp_path: Path):
    matrix = np.eye(2, dtype=np.float32)
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 3]})

    with pytest.raises(ValueError, match="out of bounds"):
        generate_filtered_series_labels(
            out_dir,
            in_series_threshold=0.5,
            out_of_series_threshold=0.5,
        )


def test_generate_filtered_series_labels_rejects_image_path_length_mismatch(tmp_path: Path):
    matrix = np.eye(2, dtype=np.float32)
    out_dir = _write_run_output(
        tmp_path,
        matrix,
        {"A": [0, 1]},
        image_paths=["/dataset/img_0.png"],
    )

    with pytest.raises(ValueError, match="length does not match pairwise distances size"):
        generate_filtered_series_labels(
            out_dir,
            in_series_threshold=0.5,
            out_of_series_threshold=0.5,
        )


def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "script",
            "--pairwise-output-dir",
            "/tmp/output",
            "--in-series-threshold",
            "0.4",
            "--out-of-series-threshold",
            "0.7",
        ],
    )

    (
        pairwise_output_dir,
        in_series_threshold,
        in_series_threshold_label,
        out_of_series_threshold,
        out_of_series_threshold_label,
    ) = parse_args()

    assert pairwise_output_dir == Path("/tmp/output").resolve()
    assert in_series_threshold == pytest.approx(0.4)
    assert in_series_threshold_label == "0.4"
    assert out_of_series_threshold == pytest.approx(0.7)
    assert out_of_series_threshold_label == "0.7"


def test_main_entry_point(tmp_path: Path):
    matrix = np.array(
        [
            [0.0, 0.1, 0.9, 1.0],
            [0.1, 0.0, 0.9, 1.0],
            [0.9, 0.9, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    out_dir = _write_run_output(tmp_path, matrix, {"A": [0, 1, 2]})

    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            "script",
            "--pairwise-output-dir",
            str(out_dir),
            "--in-series-threshold",
            "0.50",
            "--out-of-series-threshold",
            "0.50",
        ]
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.argv = original_argv

    created = sorted(out_dir.glob("filtered_series_by_avg_distance_in_t0_50_out_t0_50"))
    assert len(created) == 1
    assert json.loads((created[0] / "series_to_indices.json").read_text()) == {
        "A": [0, 1]
    }
