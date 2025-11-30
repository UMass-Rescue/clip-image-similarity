import os
from pathlib import Path

import pytest

from clip_image_similarity.utils import (
    DEFAULT_EXTS,
    Logger,
    configure_logging,
    batched,
    default_device,
    find_images,
    log,
    plural,
)


@pytest.mark.parametrize(
    ("value", "word", "expected"),
    [
        (0, "image", "0 images"),
        (1, "image", "1 image"),
        (2, "image", "2 images"),
    ],
)
def test_plural(value, word, expected):
    assert plural(value, word) == expected


@pytest.mark.parametrize(
    ("seq", "batch_size", "expected"),
    [
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([1, 2, 3], 4, [[1, 2, 3]]),
        ([], 3, []),
    ],
)
def test_batched(seq, batch_size, expected):
    assert [list(batch) for batch in batched(seq, batch_size)] == expected


@pytest.fixture(autouse=True)
def reset_logger():
    # Ensure global logger does not leak between tests
    yield
    configure_logging(None)


def test_logger_writes_to_file_and_stdout(tmp_path, capsys):
    log_path = tmp_path / "logs" / "run.log"
    logger = Logger(log_path)
    logger.log("hello")

    text = log_path.read_text(encoding="utf-8")
    assert "hello" in text

    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_global_log_respects_allow_file(tmp_path):
    log_path = tmp_path / "logs" / "global.log"
    configure_logging(log_path)
    log("write-me")
    log("skip-file", allow_file=False)

    text = log_path.read_text(encoding="utf-8")
    assert "write-me" in text
    assert "skip-file" not in text


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [
        (True, "cuda"),
        (False, "cpu"),
    ],
)
def test_default_device(monkeypatch, cuda_available, expected):
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    assert default_device() == expected


def test_find_images_recurses_and_sorts(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    files = [
        tmp_path / "a.jpg",
        tmp_path / "b.png",
        nested / "c.JPG",
        tmp_path / "ignore.txt",
    ]
    for path in files:
        path.write_text("x", encoding="utf-8")

    found = find_images(tmp_path, DEFAULT_EXTS)
    # Should return absolute, sorted paths and ignore non-image files
    assert [p.name for p in found] == ["a.jpg", "b.png", "c.JPG"]
    assert all(p.is_absolute() for p in found)
