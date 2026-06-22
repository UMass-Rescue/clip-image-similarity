import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.prepare_finetune_workflow import (
    deduplicate_label_mapping,
    generate_workflow_files,
    hamming_distance,
    phash_image,
)


def _write_image(path: Path, *, inverted: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    background = (20, 40, 60) if not inverted else (220, 220, 220)
    foreground = (220, 220, 220) if not inverted else (20, 40, 60)
    image = Image.new("RGB", (24, 24), color=background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 10, 10), fill=foreground)
    draw.line((4, 18, 22, 6), fill=foreground, width=3)
    image.save(path)
    return path.resolve()


def _write_labels(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phash_matches_identical_images(tmp_path):
    first = _write_image(tmp_path / "first.png")
    second = _write_image(tmp_path / "second.png")

    assert hamming_distance(phash_image(first), phash_image(second)) == 0


def test_deduplicate_label_mapping_removes_near_duplicate_files_and_repeated_paths(
    tmp_path,
):
    original = _write_image(tmp_path / "images" / "original.png")
    duplicate = _write_image(tmp_path / "images" / "duplicate.png")
    distinct = _write_image(tmp_path / "images" / "distinct.png", inverted=True)
    labels_path = _write_labels(
        tmp_path / "labels.json",
        {
            "series_a": [original.as_posix(), duplicate.as_posix()],
            "series_b": [original.as_posix(), distinct.as_posix()],
        },
    )
    output_path = tmp_path / "out" / "labels.json"
    manifest_path = tmp_path / "out" / "manifest.json"
    output_path.parent.mkdir()

    result = deduplicate_label_mapping(
        input_json=labels_path,
        output_json=output_path,
        manifest_json=manifest_path,
        phash_threshold=0,
    )

    assert result == {
        "series_a": [original.as_posix()],
        "series_b": [distinct.as_posix()],
    }
    assert json.loads(output_path.read_text()) == result
    manifest = json.loads(manifest_path.read_text())
    assert manifest["removed_unique_image_count"] == 1
    assert manifest["removed_image_reference_count"] == 2
    assert {item["path"] for item in manifest["removed_image_references"]} == {
        duplicate.as_posix(),
        original.as_posix(),
    }


def test_generate_workflow_files_points_configs_at_deduplicated_labels(tmp_path):
    original = _write_image(tmp_path / "images" / "original.png")
    duplicate = _write_image(tmp_path / "images" / "duplicate.png")
    labels_path = _write_labels(
        tmp_path / "labels.json",
        {"series_a": [original.as_posix(), duplicate.as_posix()]},
    )

    generated = generate_workflow_files(
        {
            "dataset_name": "dataset",
            "image_dir": tmp_path / "images",
            "labels_json": labels_path,
            "output_dir": tmp_path / "workflow",
        },
        epochs=9,
        deduplicate_images=True,
        phash_threshold=0,
    )

    assert generated["deduplicated_labels"].is_file()
    assert generated["dedupe_manifest"].is_file()
    data_config = json.loads(generated["data_config"].read_text())
    eval_config = json.loads(generated["test_eval_config"].read_text())
    deduped_labels = generated["deduplicated_labels"].as_posix()
    assert data_config["datasets"][0]["series_labels_path"] == deduped_labels
    assert eval_config["series_image_path"] == (tmp_path / "images").as_posix()
