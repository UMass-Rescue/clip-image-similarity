import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def _canonicalize_path(path_str: str) -> str:
    """Return an absolute, POSIX-style canonical path string.

    Args:
        path_str: Path to canonicalize.
    Returns:
        Canonical POSIX-style absolute path.
    """
    return Path(path_str).resolve().as_posix()


class PathIdStore:
    """
    Maintains a mapping from canonical file paths to stable numeric IDs.
    - If mapping_file is provided, loads on init (if exists) and saves on demand.
    - If mapping_file is None, operates purely in-memory.
    """

    def __init__(self, mapping_file: str | None = None):
        """Initialize the store, optionally loading from an existing mapping file."""
        self._file: Path | None = Path(mapping_file) if mapping_file else None
        self._map: dict[str, str] = {}
        self._next_id_value: int = 1
        if self._file is not None and self._file.exists():
            self._load()
        self._recompute_next_id()

    @classmethod
    def from_env(cls) -> "PathIdStore":
        path = os.getenv("ANON_ID_MAP_FILEPATH")
        if path and path.strip():
            return cls(path.strip())
        output_dir = os.getenv("OUTPUT_DIR", "./results")
        default_path = Path(output_dir) / "file_to_id_map" / "anon_id_map.json"
        return cls(str(default_path))

    @property
    def has_persistence(self) -> bool:
        return self._file is not None

    def _load(self) -> None:
        """Load persisted mapping from disk into memory."""
        try:
            with open(self._file, "r") as f:  # type: ignore[arg-type]
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str):
                        canon = _canonicalize_path(k)
                        self._map[canon] = v
        except Exception:
            # If load fails, start with empty map
            self._map = {}

    def _recompute_next_id(self) -> None:
        """Recompute the next ID value based on the current mapping values."""
        max_id = 0
        for v in self._map.values():
            try:
                n = int(v)
            except Exception:
                continue
            if n > max_id:
                max_id = n
        self._next_id_value = max_id + 1

    def _next_id(self) -> str:
        """Generate the next anonymous ID as a string."""
        value = str(self._next_id_value)
        self._next_id_value += 1
        return value

    def get_or_assign_id_for_path(self, path_str: str) -> str:
        """Return the existing anon ID for path_str or assign a new one.

        Args:
            path_str: Arbitrary path string that will be canonicalized.
        Returns:
            Stable anonymous ID for the canonicalized path.
        """
        canon = _canonicalize_path(path_str)
        if canon in self._map:
            return self._map[canon]
        new_id = self._next_id()
        self._map[canon] = new_id
        return new_id

    def get_ids_for_paths(self, paths: Iterable[str]) -> dict[str, str]:
        """Return IDs for many paths, creating new IDs as needed.

        Args:
            paths: Iterable of path strings.
        Returns:
            Mapping of original path strings to assigned anon IDs.
        """
        result: dict[str, str] = {}
        for p in paths:
            result[p] = self.get_or_assign_id_for_path(p)
        return result

    def save(self) -> None:
        """Persist the current mapping to disk if a backing file was provided."""
        if self._file is None:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        # Writing to a temporary file first to avoid corruption if the process is interrupted
        tmp_path = self._file.with_suffix(self._file.suffix + ".tmp")
        data = {k: v for k, v in self._map.items()}
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self._file)


def create_path_id_store(output_dir: Path) -> PathIdStore:
    """Create a persisted PathIdStore rooted under the given output directory.

    Args:
        output_dir: Base directory for results.
    Returns:
        PathIdStore configured to write to file_to_id_map/anon_id_map.json.
    """
    mapping_file = output_dir / "file_to_id_map" / "anon_id_map.json"
    return PathIdStore(str(mapping_file))


def build_id_map_for_paths(store: PathIdStore, paths: List[Path]) -> Dict[str, str]:
    """Resolve paths to strings and fetch/assign anon IDs with the provided store.

    Args:
        store: PathIdStore instance.
        paths: List of image paths to map.
    Returns:
        Mapping of absolute path strings to anon ID strings.
    """
    str_paths = [str(p) for p in paths]
    return store.get_ids_for_paths(str_paths)
