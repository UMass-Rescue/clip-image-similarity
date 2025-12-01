import json
from pathlib import Path
import runpy
import sys

import pytest

from clip_image_similarity.generate_anonymous_labels import (
    generate_anonymous_labels,
    parse_args,
)


def write_image_paths(output_dir: Path, paths: list) -> None:
    """Helper to write image_paths.json."""
    image_paths_file = output_dir / "image_paths.json"
    with image_paths_file.open("w", encoding="utf-8") as f:
        json.dump(paths, f)


def write_labels(labels_path: Path, payload) -> None:
    """Helper to write labels JSON."""
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_generate_anonymous_labels_happy_path(tmp_path):
    """Test successful generation of series_to_indices.json."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", "/path/to/image3.jpg"]
    write_image_paths(output_dir, image_paths)

    labels_path = tmp_path / "labels.json"
    write_labels(
        labels_path,
        {
            "series1": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
            "series2": ["/path/to/image3.jpg"],
        },
    )

    generate_anonymous_labels(output_dir, labels_path)

    series_indices_file = output_dir / "series_to_indices.json"
    assert series_indices_file.exists()

    with series_indices_file.open("r", encoding="utf-8") as f:
        result = json.load(f)

    assert result == {"series1": [0, 1], "series2": [2]}


def test_generate_anonymous_labels_missing_files(tmp_path):
    """Test errors when required files are missing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    labels_path = tmp_path / "labels.json"

    # Missing image_paths.json
    write_labels(labels_path, {"series1": []})
    with pytest.raises(FileNotFoundError, match="image_paths.json not found"):
        generate_anonymous_labels(output_dir, labels_path)

    # Missing labels file
    write_image_paths(output_dir, [])
    labels_path = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="Labels file not found"):
        generate_anonymous_labels(output_dir, labels_path)


def test_generate_anonymous_labels_overwrite_behavior(tmp_path):
    """Test overwrite protection and functionality."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    write_image_paths(output_dir, ["/path/to/image1.jpg"])

    labels_path = tmp_path / "labels.json"
    write_labels(labels_path, {"series1": ["/path/to/image1.jpg"]})

    series_indices_file = output_dir / "series_to_indices.json"
    series_indices_file.write_text('{"old": [999]}', encoding="utf-8")

    # Should fail without overwrite flag
    with pytest.raises(FileExistsError, match="already exists. Use --overwrite"):
        generate_anonymous_labels(output_dir, labels_path, overwrite=False)

    # Should succeed with overwrite flag
    generate_anonymous_labels(output_dir, labels_path, overwrite=True)

    with series_indices_file.open("r", encoding="utf-8") as f:
        result = json.load(f)

    assert result == {"series1": [0]}


def test_generate_anonymous_labels_invalid_image_paths_json(tmp_path):
    """Test error when image_paths.json is not a list."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    image_paths_file = output_dir / "image_paths.json"
    with image_paths_file.open("w", encoding="utf-8") as f:
        json.dump({"not": "a list"}, f)

    labels_path = tmp_path / "labels.json"
    write_labels(labels_path, {"series1": []})

    with pytest.raises(ValueError, match="must contain a list"):
        generate_anonymous_labels(output_dir, labels_path)


def test_generate_anonymous_labels_label_validation(tmp_path):
    """Test that label validation errors from map_labels_to_indices are propagated."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    write_image_paths(output_dir, ["/path/to/image1.jpg"])

    labels_path = tmp_path / "labels.json"
    write_labels(
        labels_path,
        {"series1": ["/path/to/image1.jpg", "/path/to/missing.jpg"]},
    )

    # Validation should catch missing paths
    with pytest.raises(ValueError, match="not found in the embedded image list"):
        generate_anonymous_labels(output_dir, labels_path)


def test_parse_args(monkeypatch):
    """Test CLI argument parsing with various flag combinations."""
    # With overwrite flag
    monkeypatch.setattr(
        "sys.argv",
        [
            "script",
            "--output-dir",
            "/tmp/output",
            "--labels",
            "/tmp/labels.json",
            "--overwrite",
        ],
    )
    output_dir, labels_path, overwrite = parse_args()
    assert output_dir == Path("/tmp/output").resolve()
    assert labels_path == Path("/tmp/labels.json").resolve()
    assert overwrite is True

    # Without overwrite flag (short flags)
    monkeypatch.setattr(
        "sys.argv",
        ["script", "-o", "/tmp/output", "-l", "/tmp/labels.json"],
    )
    output_dir, labels_path, overwrite = parse_args()
    assert output_dir == Path("/tmp/output").resolve()
    assert labels_path == Path("/tmp/labels.json").resolve()
    assert overwrite is False


def test_main_entry_point(tmp_path, monkeypatch, capsys):
    """Test main() entry point executes end-to-end including if __name__ == '__main__' block."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    write_image_paths(output_dir, ["/path/to/image1.jpg"])

    labels_path = tmp_path / "labels.json"
    write_labels(labels_path, {"series1": ["/path/to/image1.jpg"]})

    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            "script",
            "--output-dir",
            str(output_dir),
            "--labels",
            str(labels_path),
        ]

        # Run module (covers both main() and if __name__ == "__main__")
        runpy.run_module(
            "clip_image_similarity.generate_anonymous_labels", run_name="__main__"
        )

        series_indices_file = output_dir / "series_to_indices.json"
        assert series_indices_file.exists()

        captured = capsys.readouterr()
        assert "Successfully generated" in captured.out
        assert "Mapped 1 series" in captured.out

    finally:
        sys.argv = original_argv

