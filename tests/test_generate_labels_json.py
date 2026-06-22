import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_labels_json.py"
SPEC = importlib.util.spec_from_file_location("generate_labels_json", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

generate_labels_json = MODULE.generate_labels_json
normalize_extensions = MODULE.normalize_extensions


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image-ish")
    return path


def test_generate_labels_json_maps_child_directories_to_images(tmp_path):
    series_dir = tmp_path / "series"
    first = _touch(series_dir / "series_b" / "b.JPG")
    second = _touch(series_dir / "series_a" / "a.jpg")
    _touch(series_dir / "series_a" / "notes.txt")
    output_json = tmp_path / "labels.json"

    result = generate_labels_json(
        series_dir=series_dir,
        output_json=output_json,
    )

    assert result == {
        "series_a": [second.resolve().as_posix()],
        "series_b": [first.resolve().as_posix()],
    }
    assert json.loads(output_json.read_text()) == result


def test_generate_labels_json_can_write_relative_paths_and_skip_small_series(tmp_path):
    series_dir = tmp_path / "dataset" / "series"
    kept = _touch(series_dir / "kept" / "one.jpg")
    _touch(series_dir / "kept" / "two.png")
    _touch(series_dir / "skipped" / "one.jpg")
    output_json = tmp_path / "labels.json"

    result = generate_labels_json(
        series_dir=series_dir,
        output_json=output_json,
        min_images_per_series=2,
        relative_to=tmp_path / "dataset",
    )

    assert result == {
        "kept": [
            kept.relative_to(tmp_path / "dataset").as_posix(),
            "series/kept/two.png",
        ]
    }


def test_generate_labels_json_refuses_to_overwrite_by_default(tmp_path):
    series_dir = tmp_path / "series"
    _touch(series_dir / "series_a" / "a.jpg")
    output_json = tmp_path / "labels.json"
    output_json.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_labels_json(series_dir=series_dir, output_json=output_json)


def test_normalize_extensions_accepts_dotted_and_undotted_extensions():
    assert normalize_extensions(["jpg", ".PNG"]) == {".jpg", ".png"}
