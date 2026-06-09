import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.filter_loadable_labels import filter_loadable_labels


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(path)
    return path.resolve()


def _write_labels(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_filter_loadable_labels_skips_bad_images_and_drops_small_series(tmp_path):
    good_a = _write_image(tmp_path / "a.png")
    good_b = _write_image(tmp_path / "b.png")
    singleton = _write_image(tmp_path / "singleton.png")
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    labels_path = _write_labels(
        tmp_path / "labels.json",
        {
            "keep": [good_a.as_posix(), bad.as_posix(), good_b.as_posix()],
            "drop_small": [singleton.as_posix()],
            "drop_all": [bad.as_posix()],
        },
    )
    output_path = tmp_path / "out" / "loadable_series.json"
    skipped_path = tmp_path / "out" / "skipped_images.json"

    result = filter_loadable_labels(
        labels_json=labels_path,
        output_json=output_path,
        skipped_json=skipped_path,
    )

    assert result == {"keep": [good_a.as_posix(), good_b.as_posix()]}
    assert json.loads(output_path.read_text()) == result
    manifest = json.loads(skipped_path.read_text())
    assert manifest["failed_image_count"] == 1
    assert manifest["dropped_series_count"] == 2
    assert manifest["skipped_images"][0]["path"] == bad.resolve().as_posix()
    assert manifest["skipped_images"][0]["affected_series"] == ["keep", "drop_all"]


def test_filter_loadable_labels_fails_when_no_series_remain(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    labels_path = _write_labels(
        tmp_path / "labels.json", {"drop_all": [bad.as_posix()]}
    )
    output_path = tmp_path / "loadable_series.json"
    skipped_path = tmp_path / "skipped_images.json"

    with pytest.raises(RuntimeError, match="No series remain"):
        filter_loadable_labels(
            labels_json=labels_path,
            output_json=output_path,
            skipped_json=skipped_path,
        )

    assert not output_path.exists()
    assert skipped_path.exists()
