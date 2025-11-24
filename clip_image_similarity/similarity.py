from __future__ import annotations

import torch

from .utils import log


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
