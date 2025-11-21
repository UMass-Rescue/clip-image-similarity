from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from .utils import log


class SimilarityComputer:
    def __init__(self, device: str):
        """Create a SimilarityComputer that will operate on the specified device."""
        self.device = device

    def cosine_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute a full cosine similarity matrix on the chosen device.

        Args:
            embeddings: CPU tensor of shape (N, D) representing image embeddings.
        Returns:
            CPU tensor of shape (N, N) with cosine similarities.
        """
        log("Computing cosine similarity matrix...")
        x = embeddings.to(self.device)
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


def compute_similarity_and_distances(
    embeddings: torch.Tensor,
    device: str,
    image_paths: List[Path],
    id_map: Dict[str, str],
) -> List[dict]:
    """Compute full similarity and distance matrices then flatten to pairwise records.

    Args:
        embeddings: CPU embeddings tensor (N, D).
        device: Device string for similarity computation.
        image_paths: Ordered image paths.
        id_map: Mapping of image path string to anon ID string.
    Returns:
        List of pairwise distance dicts.
    """
    computer = SimilarityComputer(device=device)
    sim = computer.cosine_similarity_matrix(embeddings)
    dist = computer.similarity_to_distance(sim)
    return computer.pairwise_distances(image_paths, id_map, dist)
