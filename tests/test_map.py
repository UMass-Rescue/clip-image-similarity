import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import runpy

from metrics import map as map_mod
from clip_image_similarity.topk import TopKNeighbors


def write_npz_distances(tmp_path: Path, flat: np.ndarray) -> Path:
    path = tmp_path / "dists.npz"
    np.savez_compressed(path, distances=flat.astype(np.float32), allow_pickle=False)
    return path


def write_npz_topk(tmp_path: Path, indices: np.ndarray, distances: np.ndarray) -> Path:
    path = tmp_path / "topk.npz"
    np.savez_compressed(
        path,
        indices=indices,
        distances=distances,
        top_k=indices.shape[1],
        dtype=str(distances.dtype),
        index_dtype=str(indices.dtype),
        allow_pickle=False,
    )
    return path


def write_json(tmp_path: Path, name: str, obj) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_series_indices_happy_and_errors(tmp_path, monkeypatch):
    """Validate both the happy path and each schema validation error for series indices.

    We first write a valid JSON mapping and confirm it is returned unchanged. Then we
    deliberately violate every rule (non-dict root, non-string keys, non-list values,
    non-string paths, non-integer indices) to ensure load_series_indices raises the
    exact ValueError in each scenario.
    """
    good = write_json(tmp_path, "series.json", {"s": [0, 1]})
    assert map_mod.load_series_indices(good) == {"s": [0, 1]}

    bad = write_json(tmp_path, "bad.json", ["not", "dict"])
    with pytest.raises(ValueError, match="must be a JSON object"):
        map_mod.load_series_indices(bad)

    monkeypatch.setattr(map_mod.json, "load", lambda f: {1: []})
    with pytest.raises(ValueError, match="Series names must be strings"):
        map_mod.load_series_indices(good)

    monkeypatch.setattr(map_mod.json, "load", lambda f: {"s": "not-a-list"})
    with pytest.raises(ValueError, match="must be a list"):
        map_mod.load_series_indices(good)

    monkeypatch.setattr(map_mod.json, "load", lambda f: {"s": ["bad"]})
    with pytest.raises(ValueError, match="must be integers"):
        map_mod.load_series_indices(good)


def test_validate_series_indices(tmp_path):
    """Ensure _validate_series_indices deduplicates and rejects out-of-range indices.

    The function should return sorted de-duplicated lists when all indices are valid,
    but it must raise when any index exceeds the allowed neighbor source size; both
    behaviors are asserted here.
    """
    series = {"s": [2, 1, 1]}
    cleaned = map_mod._validate_series_indices(series, n_items=3)
    assert cleaned == {"s": [1, 2]}
    with pytest.raises(ValueError):
        map_mod._validate_series_indices({"s": [5]}, n_items=2)


def test_build_rankings_with_packed_distances():
    """Confirm packed distances produce fully sorted rankings for every node.

    Using a tiny analytical distance matrix lets us compute the expected orderings
    by hand; the test ensures build_rankings reproduces exactly those sequences.
    """
    # Distances for pairs: (0,1)=1, (0,2)=2, (1,2)=3
    flat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dist = map_mod.PackedDistances(flat)
    rankings = map_mod.build_rankings(dist, {0, 1, 2}, {0, 1, 2})
    assert rankings[0] == [1, 2]
    assert rankings[1] == [0, 2]
    assert rankings[2] == [0, 1]


def test_build_rankings_with_topk_and_bounds():
    """Exercise TopK path, candidate filtering, and query bounds validation.

    We construct deterministic top-k neighbors, validate their rankings, ensure
    filtering drops disallowed neighbors, and finally confirm building rankings
    with an out-of-range query raises ValueError as advertised.
    """
    indices = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
        ],
        dtype=np.uint16,
    )
    distances = np.array(
        [
            [0.3, 0.8],
            [0.4, 0.9],
            [0.2, 0.7],
        ],
        dtype=np.float32,
    )
    topk = TopKNeighbors(indices, distances, top_k=2, dtype="float32")
    rankings = map_mod.build_rankings(topk, {0, 1, 2}, {0, 1, 2})
    assert rankings == {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    # Candidate filtering should drop neighbors not in the allowed set.
    filtered = map_mod.build_rankings(topk, {2}, {0, 1})
    assert filtered == {0: [2], 1: [2]}

    with pytest.raises(ValueError):
        map_mod.build_rankings(topk, {0, 1, 2}, {3})


@pytest.mark.parametrize(
    (
        "case_name",
        "preds",
        "positives",
        "k",
        "expected_ap",
    ),
    [
        (
            "two_hits_top3",
            [1, 2, 3],
            {2, 3},
            3,
            (1 / 2 + 2 / 3) / 2,
        ),
        (
            "no_predictions",
            [],
            set(),
            3,
            0.0,
        ),
        (
            "cutoff_before_hit",
            [1, 2],
            {3},
            1,
            0.0,
        ),
        (
            "single_positive_late_hit",
            [0, 1, 2],
            {2},
            5,
            1 / 3,
        ),
    ],
)
def test_average_precision(case_name, preds, positives, k, expected_ap):
    """Detailed AP@k breakdowns for varied ranking situations.

    two_hits_top3:
      - Positives {2,3} appear at ranks 2 and 3, giving precisions 1/2 and 2/3.
      - denom=min(k, |positives|)=2, so AP = ((1/2)+(2/3))/2.

    no_predictions:
      - Either no predictions or no positives → hits never occur → AP=0.

    cutoff_before_hit:
      - k=1 limits us to the first prediction, which misses the only positive.
      - denom=min(1,1)=1 and hits=0, so AP=0.

    single_positive_late_hit:
      - Only target is index 2, retrieved at rank 3.
      - denom=min(5,1)=1, so AP equals precision at that hit: 1/3.
    """

    assert map_mod.average_precision_at_k(preds, positives, k=k) == pytest.approx(expected_ap)


def test_mean_average_precision_partial_recall():
    """Explain a mixed-quality mAP@k computation for two queries.

    Query 0: predictions [1,2]; only positive is 1, found at rank 1 → AP=1.0.
    Query 1: predictions [2,0]; positive is 0, found at rank 2 → precision 1/2, denom=1 → AP=0.5.
    Averaging APs for queries {0,1} yields mean average precision (1.0 + 0.5) / 2 = 0.75.
    """

    preds_by_query = {0: [1, 2], 1: [2, 0]}
    positives_lookup = {0: {1}, 1: {0}}
    result = map_mod.mean_average_precision_at_k(preds_by_query, {0, 1}, k=2, positives_lookup=positives_lookup)
    assert result == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("case_name", "rankings", "series_to_indices", "max_k", "expected"),
    [
        (
            "two_images_k1",
            {0: [1], 1: [0]},
            {"s": [0, 1]},
            1,
            {"s": [1.0]},
        ),
        (
            "three_images_k3_partial",
            {
                0: [3, 2, 1],
                1: [3, 0, 2],
                2: [3, 0, 1],
                3: [0, 1, 2],
            },
            {"s": [0, 1, 2]},
            3,
            {"s": [0.0, 0.25, 7 / 12]},
        ),
    ],
)
def test_compute_series_map_and_errors(case_name, rankings, series_to_indices, max_k, expected):
    """Ensure compute_series_map produces intuitive mAP@k curves for multiple scenarios.

    Two-image scenario (map@1=1.0):
      - Each query has one relevant partner and the rankings list it first.
      - At k=1 we have denom=min(k, len(positives))=min(1, 1)=1, precision is 1/1 at rank 1, so AP per query is 1.0.
      - Averaging across both queries yields map@1=1.0.

    Three-image scenario (map@1=0, map@2=0.25, map@3=7/12):
      - Positives per query are the other two series members; rank 3 is an unrelated distractor placed first.
      - k=1: denom=min(1, 2)=1, top prediction misses, so AP=0 for each query → map@1=0.
      - k=2: first hit occurs at rank 2, giving precision 1/2; denom=min(2, 2)=2 → AP=(1/2)/2=0.25.
      - k=3: hits at ranks 2 and 3 yield precisions 1/2 and 2/3; AP=(1/2+2/3)/2=7/12 → map@3=7/12.
    """

    result = map_mod.compute_series_map(series_to_indices, rankings, max_k=max_k)
    assert set(result.keys()) == set(expected.keys())
    for series, values in expected.items():
        assert len(result[series]) == len(values)
        for got, want in zip(result[series], values):
            assert got == pytest.approx(want)

    with pytest.raises(ValueError):
        map_mod.compute_series_map({"s": [0]}, rankings, max_k=1)


def test_write_series_map_csv(tmp_path):
    """Ensure CSV output encodes each series' map@k plus the averaged mean row.

    After writing the CSV we parse it back, confirm headers, check each series row
    matches the formatted decimals, and verify the computed mean values equal the
    manual averages the writer is supposed to produce.
    """
    series_map = {"s": [0.5, 1.0], "t": [1.0, 0.0]}
    out = tmp_path / "map.csv"
    map_mod.write_series_map_csv(series_map, out)
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == ["series", "map@1", "map@2"]
    rows_by_series = {row["series"]: row for row in rows}
    assert rows_by_series["s"]["map@1"] == f"{0.5:.6f}"
    assert rows_by_series["s"]["map@2"] == f"{1.0:.6f}"
    assert rows_by_series["t"]["map@1"] == f"{1.0:.6f}"
    assert rows_by_series["t"]["map@2"] == f"{0.0:.6f}"
    assert rows_by_series["mean"]["map@1"] == f"{(0.5 + 1.0) / 2:.6f}"
    assert rows_by_series["mean"]["map@2"] == f"{(1.0 + 0.0) / 2:.6f}"


def test_main_with_distances_and_series_indices(tmp_path, monkeypatch):
    """Smoke test CLI path using packed distances with provided series indices.

    Passing minimal arguments plus valid artifacts should succeed and produce the
    requested CSV; failure here would indicate regressions in argument parsing or
    the packed-distance mAP workflow.
    """
    flat = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # n=3
    dist_path = write_npz_distances(tmp_path, flat)
    series_path = write_json(tmp_path, "series.json", {"s": [0, 1]})
    out_csv = tmp_path / "out.csv"
    argv = [
        "prog",
        "--distances",
        str(dist_path),
        "--series-indices",
        str(series_path),
        "--output_csv",
        str(out_csv),
    ]
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(sys, "argv", argv)
    map_mod.main()
    assert out_csv.exists()


def test_main_with_topk_and_labels(tmp_path, monkeypatch):
    """Smoke test CLI path using top-k neighbors with labels and image paths.

    We provide consistent labels/image-paths derived from the mocked dataset and
    expect the CLI to finish and emit the CSV. Any mismatch would surface issues
    in label mapping or top-k loading logic.
    """
    # Build three image paths and labels
    imgs = []
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        p = tmp_path / name
        p.write_text("x", encoding="utf-8")
        imgs.append(p)
    labels_path = write_json(tmp_path, "labels.json", {"s": [p.as_posix() for p in imgs[:2]]})
    image_paths_path = write_json(tmp_path, "image_paths.json", [p.as_posix() for p in imgs])
    indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.uint16)
    distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.4]], dtype=np.float32)
    topk_path = write_npz_topk(tmp_path, indices, distances)
    out_csv = tmp_path / "topk.csv"
    argv = [
        "prog",
        "--topk",
        str(topk_path),
        "--labels",
        str(labels_path),
        "--image-paths",
        str(image_paths_path),
        "--output_csv",
        str(out_csv),
        "--labeled-images-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    map_mod.main()
    assert out_csv.exists()


def test_main_requires_distances_or_topk(tmp_path, monkeypatch):
    """Verify CLI enforcement that exactly one of --distances/--topk is supplied.

    Running without either input should immediately raise ValueError, preventing a
    confusing runtime failure deeper in the pipeline.
    """
    out_csv = tmp_path / "out.csv"
    argv = ["prog", "--output_csv", str(out_csv)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        map_mod.main()


def test_main_requires_labels_and_image_paths(tmp_path, monkeypatch):
    """Ensure CLI enforces providing labels and image paths whenever --topk is used.

    When top-k neighbors are supplied without the auxiliary label/image metadata,
    the CLI should refuse to run because it cannot map series to indices, so we
    assert that ValueError is raised.
    """
    topk_path = write_npz_topk(
        tmp_path,
        np.array([[1]], dtype=np.uint16),
        np.array([[0.1]], dtype=np.float32),
    )
    out_csv = tmp_path / "out.csv"
    argv = [
        "prog",
        "--topk",
        str(topk_path),
        "--output_csv",
        str(out_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        map_mod.main()


def test_main_topk_too_small_errors(tmp_path, monkeypatch):
    """Check CLI failure when provided top-k neighbors cannot cover the largest series.

    Here the largest series has 3 images but the stored top-k results only keep one
    neighbor per row; map.main should detect that positives will be missing and abort.
    """
    indices = np.array([[1], [0], [0]], dtype=np.uint16)  # k=1
    distances = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    topk_path = write_npz_topk(tmp_path, indices, distances)
    series_path = write_json(tmp_path, "series.json", {"s": [0, 1, 2]})
    out_csv = tmp_path / "out.csv"
    argv = [
        "prog",
        "--topk",
        str(topk_path),
        "--series-indices",
        str(series_path),
        "--output_csv",
        str(out_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        map_mod.main()


def test_map_entrypoint_runpy(tmp_path, monkeypatch):
    """Cover execution via runpy so the __main__ guard stays wired correctly.

    By running metrics.map as a module we ensure the argument parsing plus run()
    sequence operates when __name__ == "__main__", mirroring the CLI entrypoint.
    """
    # Execute module as __main__ to cover entrypoint guard.
    flat = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    dist_path = write_npz_distances(tmp_path, flat)
    series_path = write_json(tmp_path, "series.json", {"s": [0, 1]})
    out_csv = tmp_path / "entry.csv"
    argv = [
        "metrics.map",
        "--distances",
        str(dist_path),
        "--series-indices",
        str(series_path),
        "--output_csv",
        str(out_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    runpy.run_module("metrics.map", run_name="__main__")
    assert out_csv.exists()
