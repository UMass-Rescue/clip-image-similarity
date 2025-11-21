from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from tqdm.auto import tqdm

from .utils import log
from . import parquet_writer


class SimilarityComputer:
    def __init__(self, device: str):
        """Create a SimilarityComputer that will operate on the specified device."""
        self.device = device

    def cosine_similarity_matrix(
        self, embeddings: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Compute a full cosine similarity matrix on the chosen device.

        Args:
            embeddings: CPU tensor of shape (N, D) representing image embeddings.
            dtype: Torch dtype for similarity/distances (e.g., torch.float32 or torch.float16).
        Returns:
            CPU tensor of shape (N, N) with cosine similarities.
        """
        log("Computing cosine similarity matrix...")
        x = embeddings.to(self.device, dtype=dtype)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        sim = x @ x.T
        log("Similarity matrix computed.")
        return sim.cpu()

    @staticmethod
    def similarity_to_distance(sim_matrix: torch.Tensor) -> torch.Tensor:
        """Convert cosine similarity values into distances.

        Args:
            sim_matrix: Cosine similarity matrix.
        Returns:
            Distance matrix computed as 1 - sim_matrix.
        """
        return 1.0 - sim_matrix

    @staticmethod
    def stream_upper_triangle_to_parquet(
        dist_matrix: torch.Tensor,
        ids: Sequence[str],
        output_path: Path,
        *,
        chunk_size_pairs: int = 5_000_000,
        compression: str | None = None,
        compression_level: int | None = None,
    ) -> None:
        """Stream upper-triangular pairwise distances to a Parquet file.

        Args:
            dist_matrix: Square distance matrix (N, N), can be on CPU or GPU.
            ids: Ordered list of anon IDs aligned with matrix indices.
            output_path: Destination Parquet path.
            chunk_size_pairs: Number of pairs per chunk when writing.
            compression: Parquet compression codec (e.g., 'zstd', None for no compression).
            compression_level: Optional compression level for the codec (e.g., zstd level).
        """
        n = dist_matrix.shape[0]
        if dist_matrix.shape[1] != n:
            raise ValueError("Distance matrix must be square.")
        if len(ids) != n:
            raise ValueError("Length of ids must match distance matrix dimension.")
        device = dist_matrix.device
        idx_i_full, idx_j_full = torch.triu_indices(n, n, offset=1, device=device)
        total_pairs = idx_i_full.numel()

        def _iter_chunks():
            for start in range(0, total_pairs, chunk_size_pairs):
                end = min(start + chunk_size_pairs, total_pairs)
                idx_i_chunk = idx_i_full[start:end]
                idx_j_chunk = idx_j_full[start:end]
                dist_chunk = dist_matrix[idx_i_chunk, idx_j_chunk].cpu().tolist()
                yield idx_i_chunk.cpu().tolist(), idx_j_chunk.cpu().tolist(), dist_chunk

        parquet_writer.write_pairwise_parquet(
            idx_iter=_iter_chunks(),
            ids=ids,
            output_path=output_path,
            compression=compression,
            compression_level=compression_level,
            chunk_size=None,
        )

    def pairwise_distances(
        self,
        image_paths: List[Path],
        id_map: Dict[str, str],
        dist_matrix: torch.Tensor,
    ) -> List[dict]:
        """Build a list of pairwise distance records for all i<j combinations.

        Args:
            image_paths: Ordered list of image paths corresponding to distance rows/cols.
            id_map: Mapping of absolute path strings to anon IDs.
            dist_matrix: Square distance matrix (N, N).
        Returns:
            List of dicts with keys image1, image2, and distance.
        """
        n = len(image_paths)
        if dist_matrix.shape != (n, n):
            raise ValueError("Distance matrix shape does not match number of images.")

        total_pairs = n * (n - 1) // 2
        log(f"Collecting {total_pairs} pairwise distances...")
        results: List[dict] = []

        with tqdm(total=total_pairs, desc="Pairs", unit="pair") as pbar:
            for i in range(n):
                id_i = id_map[str(image_paths[i])]
                for j in range(i + 1, n):
                    id_j = id_map[str(image_paths[j])]
                    dist = float(dist_matrix[i, j])
                    results.append({"image1": id_i, "image2": id_j, "distance": dist})
                pbar.update(n - i - 1)

        log("Pairwise distance collection complete.")
        return results
