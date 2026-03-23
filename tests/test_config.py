import pytest
from pathlib import Path

from clip_image_similarity.config import (
    RunConfig,
    _normalize_exts,
    _validate_input_dir,
)


def test_validate_input_dir_rejects_missing(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ValueError):
        _validate_input_dir(missing)


def test_normalize_exts_lower_and_dedup():
    exts = (".JPG", "png", "JPG", "bmp", "PNG")
    assert _normalize_exts(exts) == (".jpg", ".png", ".bmp")


def test_runconfig_normalizes_fields(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    labels = tmp_path / "labels.json"
    labels.touch()
    checkpoint = tmp_path / "epoch_4.pt"
    checkpoint.touch()

    cfg = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_id="model",
        batch_size=8,
        device="cpu",
        image_exts=(".JPG", "png"),
        pretrained="dfn5b",
        checkpoint_path=checkpoint,
        pairwise_dtype="FLOAT16",
        top_k=5,
        labels_path=labels,
        overwrite=True,
    )

    assert cfg.input_dir == input_dir
    assert cfg.output_dir == output_dir.resolve()
    assert cfg.image_exts == (".jpg", ".png")
    assert cfg.pretrained == "dfn5b"
    assert cfg.checkpoint_path == checkpoint.resolve()
    assert cfg.pairwise_dtype == "float16"
    assert cfg.top_k == 5
    assert cfg.labels_path == labels.resolve()


@pytest.mark.parametrize("dtype", ["float8", "fp32"])
def test_runconfig_invalid_dtype(dtype, tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    with pytest.raises(ValueError):
        RunConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            model_id="model",
            batch_size=1,
            device="cpu",
            image_exts=(".jpg",),
            pairwise_dtype=dtype,
        )


@pytest.mark.parametrize("top_k", [0, -3])
def test_runconfig_invalid_topk(top_k, tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    with pytest.raises(ValueError):
        RunConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            model_id="model",
            batch_size=1,
            device="cpu",
            image_exts=(".jpg",),
            top_k=top_k,
        )


def test_runconfig_ensure_output_dir_creates(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out" / "nested"
    cfg = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_id="model",
        batch_size=1,
        device="cpu",
        image_exts=(".jpg",),
    )
    assert not output_dir.exists()
    cfg.ensure_output_dir()
    assert output_dir.exists()
