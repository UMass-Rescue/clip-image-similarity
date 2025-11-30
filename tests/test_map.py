import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
    series = {"s": [2, 1, 1]}
    cleaned = map_mod._validate_series_indices(series, n_items=3)
    assert cleaned == {"s": [1, 2]}
    with pytest.raises(ValueError):
        map_mod._validate_series_indices({"s": [5]}, n_items=2)


def test_build_rankings_with_packed_distances():
    # Distances for pairs: (0,1)=1, (0,2)=2, (1,2)=3
    flat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dist = map_mod.PackedDistances(flat)
    rankings = map_mod.build_rankings(dist, {0, 1, 2}, {0, 1, 2})
    assert rankings[0] == [1, 2]
    assert rankings[1] == [0, 2]
    assert rankings[2] == [0, 1]


def test_build_rankings_with_topk_and_bounds():
    indices = np.array([[1], [0]], dtype=np.uint16)
    distances = np.array([[0.5], [0.6]], dtype=np.float32)
    topk = TopKNeighbors(indices, distances, top_k=1, dtype="float32")
    rankings = map_mod.build_rankings(topk, {0, 1}, {0, 1})
    assert rankings == {0: [1], 1: [0]}
    with pytest.raises(ValueError):
        map_mod.build_rankings(topk, {0, 1}, {2})


def test_average_precision_and_map():
    preds = [1, 2, 3]
    positives = {2, 3}
    assert map_mod.average_precision_at_k(preds, positives, k=3) == pytest.approx(
        (1 / 2 + 2 / 3) / 2
    )
    assert map_mod.average_precision_at_k([], set(), k=3) == 0.0
    # k cutoff branch when no positives found
    assert map_mod.average_precision_at_k([1, 2], {3}, k=1) == 0.0

    preds_by_query = {0: [1, 2], 1: [0, 2]}
    positives_lookup = {0: {1}, 1: {0}}
    m = map_mod.mean_average_precision_at_k(preds_by_query, {0, 1}, k=2, positives_lookup=positives_lookup)
    assert m == pytest.approx(1.0)


def test_compute_series_map_and_errors():
    rankings = {0: [1], 1: [0]}
    series_to_indices = {"s": [0, 1]}
    result = map_mod.compute_series_map(series_to_indices, rankings, max_k=1)
    assert result == {"s": [1.0]}

    with pytest.raises(ValueError):
        map_mod.compute_series_map({"s": [0]}, rankings, max_k=1)


def test_write_series_map_csv(tmp_path):
    series_map = {"s": [0.5, 1.0], "t": [1.0, 0.0]}
    out = tmp_path / "map.csv"
    map_mod.write_series_map_csv(series_map, out)
    contents = out.read_text()
    assert "series,map@1,map@2" in contents.replace(" ", "")
    assert "mean" in contents


def test_main_with_distances_and_series_indices(tmp_path, monkeypatch):
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
    out_csv = tmp_path / "out.csv"
    argv = ["prog", "--output_csv", str(out_csv)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        map_mod.main()


def test_main_requires_labels_and_image_paths(tmp_path, monkeypatch):
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
    import runpy

    runpy.run_module("metrics.map", run_name="__main__")
    assert out_csv.exists()
