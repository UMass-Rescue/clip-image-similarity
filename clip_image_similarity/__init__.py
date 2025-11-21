"""CLIP pairwise evaluation package providing a CLI and library for computing pairwise CLIP image distances."""

from .config import RunConfig
from .cli import run

__all__ = ["RunConfig", "run"]
