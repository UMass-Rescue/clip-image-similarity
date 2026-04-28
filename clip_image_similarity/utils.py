from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Collection, Iterable, List, TypeVar

import torch

T = TypeVar("T")
_timestamp_format = "%H:%M:%S"


class Logger:
    """Simple logger that writes to stdout and optionally appends to a file."""

    def __init__(self, log_file: Path | None = None):
        self.log_file = log_file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str, allow_file: bool = True) -> None:
        line = f"[{datetime.now().strftime(_timestamp_format)}] {msg}"
        print(line, flush=True)
        if allow_file and self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


_LOGGER: Logger = Logger()


def configure_logging(log_file: Path | None) -> None:
    """Configure the global logger to also append to the provided file."""
    global _LOGGER
    _LOGGER = Logger(log_file)


def batched(seq: List[T], batch_size: int) -> Iterable[List[T]]:
    """Yield successive slices of the list for batching.

    Args:
        seq: List of items to batch.
        batch_size: Maximum number of items per batch.
    Returns:
        An iterator over list chunks of size up to batch_size.
    """
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def default_device() -> str:
    """Pick a sensible default compute device.

    Returns:
        'cuda' if available, otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str, *, allow_file: bool = True) -> None:
    """Print a timestamped log line (and optionally write to the configured file).

    Args:
        msg: Message to log.
        allow_file: If False, skip writing this message to the log file even if configured.
    """
    _LOGGER.log(msg, allow_file=allow_file)


def plural(value: int, word: str) -> str:
    """Return a simple pluralized phrase.

    Args:
        value: Count to include.
        word: Singular word to pluralize.
    Returns:
        A user-friendly phrase such as '3 images'.
    """
    suffix = "" if value == 1 else "s"
    return f"{value} {word}{suffix}"


# Common image handling utilities
DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")


def find_images(root: Path, valid_exts: Collection[str]) -> List[Path]:
    """Recursively collect image files under root.

    Args:
        root: Directory to search.
        valid_exts: Allowed file extensions (case-insensitive).
    Returns:
        Sorted list of absolute Paths to matching images.
    """
    valid = {ext.lower() for ext in valid_exts}
    image_paths: List[Path] = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid:
            image_paths.append(path.resolve())

    image_paths.sort()
    return image_paths
