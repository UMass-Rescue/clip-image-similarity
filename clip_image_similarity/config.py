from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _validate_input_dir(path: Path) -> Path:
    """Validate that path is an existing directory.

    Args:
        path: Candidate input directory.
    Returns:
        The same Path if valid.
    Raises:
        ValueError: If the path does not exist or is not a directory.
    """
    if not path.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {path}")
    return path


def _normalize_exts(exts: Tuple[str, ...]) -> Tuple[str, ...]:
    """Normalize extensions to lowercase dot-prefixed strings and deduplicate.

    Args:
        exts: Tuple of extensions, with or without leading dots.
    Returns:
        Tuple of unique, lowercase, dot-prefixed extensions preserving first occurrence order.
    """
    seen: set[str] = set()
    normalized = []
    for ext in exts:
        clean = ext if ext.startswith(".") else f".{ext}"
        clean = clean.lower()
        if clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return tuple(normalized)


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
    model_id: str
    batch_size: int
    device: str
    image_exts: Tuple[str, ...]
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Normalize paths and extensions after dataclass initialization."""
        self.input_dir = _validate_input_dir(self.input_dir)
        self.output_dir = self.output_dir.resolve()
        self.image_exts = _normalize_exts(self.image_exts)

    def ensure_output_dir(self) -> None:
        """Create the output directory and parents if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
