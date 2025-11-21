from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import RunConfig
from .utils import log


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Any, path: Path) -> None:
    """Serialize an object to JSON on disk.

    Args:
        obj: JSON-serializable object.
        path: File path where JSON should be written.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    log(f"Saved JSON to {path}")


def save_config(config: RunConfig, output_dir: Path) -> None:
    """Persist a RunConfig snapshot.

    Args:
        config: Run configuration to serialize.
        output_dir: Directory where config.json will be written.
    """
    cfg_path = output_dir / "config.json"
    save_json(_to_jsonable(asdict(config)), cfg_path)
