from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


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
    pairwise_dtype: str = "float32"  # float32|float16 for storage
    top_k: Optional[int] = None  # optional top-k sparse output
    labels_path: Optional[Path] = None  # optional labels to map to indices
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Normalize paths and extensions after dataclass initialization."""
        self.input_dir = _validate_input_dir(self.input_dir)
        self.output_dir = self.output_dir.resolve()
        self.image_exts = _normalize_exts(self.image_exts)
        self.pairwise_dtype = self.pairwise_dtype.lower()
        if self.pairwise_dtype not in {"float32", "float16"}:
            raise ValueError("pairwise_dtype must be 'float32' or 'float16'")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive if provided")
        if self.labels_path:
            self.labels_path = self.labels_path.resolve()

    def ensure_output_dir(self) -> None:
        """Create the output directory and parents if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
