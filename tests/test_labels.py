import json
from pathlib import Path

import pytest

from clip_image_similarity.labels import map_labels_to_indices


def write_labels(tmp_path: Path, payload) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_map_labels_happy_path(tmp_path):
    images = [tmp_path / "a.jpg", tmp_path / "dir" / "b.png"]
    for p in images:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    labels_path = write_labels(
        tmp_path,
        {"series1": [str(images[0]), str(images[1])], "series2": [str(images[1])]},
    )
    result = map_labels_to_indices(labels_path, images)
    assert result == {"series1": [0, 1], "series2": [1]}


def test_map_labels_rejects_non_object(tmp_path):
    labels_path = write_labels(tmp_path, ["not", "an", "object"])
    with pytest.raises(ValueError, match="Labels file must be a JSON object"):
        map_labels_to_indices(labels_path, [])


def test_map_labels_rejects_non_string_series(tmp_path, monkeypatch):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text("{}", encoding="utf-8")
    # Force json.load to return a dict with a non-string key
    monkeypatch.setattr("json.load", lambda f: {1: []})
    with pytest.raises(ValueError, match="Series names must be strings"):
        map_labels_to_indices(labels_path, [])


def test_map_labels_rejects_non_list_values(tmp_path):
    labels_path = write_labels(tmp_path, {"series": "not-a-list"})
    with pytest.raises(ValueError, match="must be a list"):
        map_labels_to_indices(labels_path, [])


def test_map_labels_rejects_non_string_paths(tmp_path):
    labels_path = write_labels(tmp_path, {"series": [123]})
    with pytest.raises(ValueError, match="must be strings"):
        map_labels_to_indices(labels_path, [])


def test_map_labels_errors_on_missing_paths(tmp_path):
    present = tmp_path / "a.jpg"
    present.write_text("x", encoding="utf-8")
    missing_path = tmp_path / "missing.jpg"
    labels_path = write_labels(
        tmp_path, {"series": [str(present), str(missing_path)]}
    )
    with pytest.raises(ValueError) as exc:
        map_labels_to_indices(labels_path, [present])
    assert "missing.jpg" in str(exc.value)
