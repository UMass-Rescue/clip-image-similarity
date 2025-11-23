from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def _build_chunk_arrays(
    idx_i: Sequence[int],
    idx_j: Sequence[int],
    distances: Sequence[float],
    ids: Sequence[str],
    dist_type: pa.DataType,
) -> pa.Table:
    """Construct a pyarrow.Table for a chunk of pairwise distances.

    Args:
        idx_i: Row indices (1D sequence).
        idx_j: Column indices (1D sequence).
        distances: Distance values aligned with idx_i/idx_j.
        ids: Global list of anon IDs indexed by integer position.
        dist_type: Arrow data type for the distance column.
    Returns:
        pyarrow.Table with columns image1, image2, distance.
    """
    image1_arr = pa.array([ids[i] for i in idx_i], type=pa.string())
    image2_arr = pa.array([ids[j] for j in idx_j], type=pa.string())
    dist_arr = pa.array(distances, type=dist_type)
    return pa.table({"image1": image1_arr, "image2": image2_arr, "distance": dist_arr})


def write_pairwise_parquet(
    idx_iter: Iterable[Tuple[Sequence[int], Sequence[int], Sequence[float]]],
    ids: Sequence[str],
    output_path: Path,
    compression: str | None = "zstd",
    compression_level: int | None = None,
    chunk_size: int | None = None,
    dist_type: pa.DataType = pa.float32(),
) -> None:
    """Stream pairwise distances to a Parquet file with optional compression.

    Args:
        idx_iter: Iterable yielding tuples (idx_i, idx_j, distances) for each chunk.
        ids: Ordered list of anon IDs; position aligns with embedding index.
        output_path: Destination Parquet path.
        compression: Parquet compression codec (e.g., 'zstd', 'gzip', or None for no compression). Defaults to zstd.
        compression_level: Optional codec-specific compression level (e.g., zstd level). Ignored if compression is None.
        chunk_size: Optional row group size; if not provided, Arrow defaults apply.
        dist_type: Arrow data type for the distance column (e.g., pa.float32(), pa.float16()).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    w_kwargs = {}
    if compression is not None:
        w_kwargs["compression"] = compression
        if compression_level is not None:
            w_kwargs["compression_level"] = compression_level
    try:
        for idx_i, idx_j, dist in idx_iter:
            table = _build_chunk_arrays(idx_i, idx_j, dist, ids, dist_type=dist_type)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, **w_kwargs)
            writer.write_table(table, row_group_size=chunk_size)
    finally:
        if writer is not None:
            writer.close()
