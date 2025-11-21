from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm.auto import tqdm
import open_clip

from .utils import batched, log, plural


class ClipEmbedder:
    def __init__(self, model_id: str, device: str):
        """Construct a ClipEmbedder bound to a model id and device."""
        self.model_id = model_id
        self.device = device
        self.model, self.preprocess = self._load_model()

    def _load_model(self):
        """Load CLIP model and preprocess from HF Hub and move to device.

        Returns:
            Tuple of (model, preprocess transform).
        """
        log(f"Loading model '{self.model_id}' on {self.device}...")
        model, preprocess = open_clip.create_model_from_pretrained(self.model_id)
        model.to(self.device)
        model.eval()
        log("Model ready for inference.")
        return model, preprocess

    def embed_paths(self, image_paths: List[Path], batch_size: int) -> torch.Tensor:
        """Compute embeddings for the provided image paths.

        Args:
            image_paths: List of image file paths to encode in order.
            batch_size: Number of images to process per batch.
        Returns:
            CPU tensor of shape (N, D) with embeddings in the same order as image_paths.
        """
        all_embs = []
        total = len(image_paths)
        log(f"Encoding {plural(total, 'image')} with batch size {batch_size}.")

        with torch.no_grad():
            for batch_paths in tqdm(batched(image_paths, batch_size), total=(total + batch_size - 1) // batch_size, desc="Embedding", unit="batch"):
                images = [self._load_image(p) for p in batch_paths]
                batch = torch.stack(images, dim=0).to(self.device)
                feats = self.model.encode_image(batch)
                all_embs.append(feats.float().cpu())

        embeddings = torch.cat(all_embs, dim=0)
        log("Finished embedding all images.")
        return embeddings

    def cleanup(self) -> None:
        """Release model resources and clear CUDA cache if applicable."""
        if hasattr(self, "model"):
            del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        log("Released model resources.")

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single image file.

        Args:
            path: Image path to open.
        Returns:
            Preprocessed tensor ready for model encoding.
        """
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.preprocess(image)


def compute_image_embeddings(image_paths: List[Path], model_id: str, device: str, batch_size: int) -> torch.Tensor:
    """Compute embeddings for image_paths with the specified model id.

    Args:
        image_paths: Image files to embed.
        model_id: HF Hub model identifier for OpenCLIP.
        device: Device string such as 'cuda' or 'cpu'.
        batch_size: Batch size for encoding.
    Returns:
        CPU tensor containing embeddings for all images in order.
    """
    embedder = ClipEmbedder(model_id=model_id, device=device)
    try:
        return embedder.embed_paths(image_paths=image_paths, batch_size=batch_size)
    finally:
        embedder.cleanup()
