from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm.auto import tqdm
import open_clip

from .image_loader import load_rgb_image
from .utils import log, plural


@dataclass(frozen=True)
class SkippedImage:
    path: Path
    index: int
    error_type: str
    error_message: str


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: torch.Tensor
    image_paths: List[Path]
    skipped_images: List[SkippedImage]


class ClipEmbedder:
    def __init__(
        self,
        model_id: str,
        device: str,
        pretrained: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """Construct a ClipEmbedder bound to a model id and device."""
        self.model_id = model_id
        self.device = device
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.model, self.preprocess = self._load_model()

    def _load_model(self):
        """Load CLIP model and preprocess from HF Hub and move to device.

        Returns:
            Tuple of (model, preprocess transform).
        """
        log(f"Loading model '{self.model_id}' on {self.device}...")
        if self.pretrained:
            model, preprocess = open_clip.create_model_from_pretrained(
                self.model_id,
                pretrained=self.pretrained,
                device=self.device,
            )
        else:
            model, preprocess = open_clip.create_model_from_pretrained(self.model_id)
            model.to(self.device)

        if self.checkpoint_path:
            log(f"Loading checkpoint from {self.checkpoint_path}...")
            open_clip.load_checkpoint(model, str(self.checkpoint_path))
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
        return self.embed_paths_with_metadata(
            image_paths=image_paths, batch_size=batch_size
        ).embeddings

    def embed_paths_with_metadata(
        self, image_paths: List[Path], batch_size: int
    ) -> EmbeddingResult:
        """Compute embeddings and report any images that could not be loaded."""
        all_embs = []
        embedded_paths: List[Path] = []
        skipped_images: List[SkippedImage] = []
        total = len(image_paths)
        log(f"Encoding {plural(total, 'image')} with batch size {batch_size}.")

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, total, batch_size),
                total=(total + batch_size - 1) // batch_size,
                desc="Embedding",
                unit="batch",
            ):
                batch_paths = image_paths[batch_start : batch_start + batch_size]
                images = []
                loaded_batch_paths = []
                for offset, path in enumerate(batch_paths):
                    image_index = batch_start + offset
                    try:
                        images.append(self._load_image(path))
                    except Exception as exc:  # image load/preprocess failure
                        skipped_images.append(
                            SkippedImage(
                                path=path,
                                index=image_index,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                        )
                        log(
                            "Skipping image "
                            f"{path}: {type(exc).__name__}: {exc}"
                        )
                        continue
                    loaded_batch_paths.append(path)
                if not images:
                    continue
                batch = torch.stack(images, dim=0).to(self.device)
                feats = self.model.encode_image(batch)
                all_embs.append(feats.float().cpu())
                embedded_paths.extend(loaded_batch_paths)

        if all_embs:
            embeddings = torch.cat(all_embs, dim=0)
        else:
            embeddings = torch.empty((0, 0), dtype=torch.float32)
        log("Finished embedding all images.")
        return EmbeddingResult(
            embeddings=embeddings,
            image_paths=embedded_paths,
            skipped_images=skipped_images,
        )

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
        image = load_rgb_image(path)
        return self.preprocess(image)


def compute_image_embeddings(
    image_paths: List[Path],
    model_id: str,
    device: str,
    batch_size: int,
    pretrained: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
) -> torch.Tensor:
    """Compute embeddings for image_paths with the specified model id.

    Args:
        image_paths: Image files to embed.
        model_id: HF Hub model identifier for OpenCLIP.
        device: Device string such as 'cuda' or 'cpu'.
        batch_size: Batch size for encoding.
        pretrained: Optional OpenCLIP pretrained identifier.
        checkpoint_path: Optional local checkpoint loaded after model creation.
    Returns:
        CPU tensor containing embeddings for all images in order.
    """
    embedder = ClipEmbedder(
        model_id=model_id,
        device=device,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )
    try:
        return embedder.embed_paths(image_paths=image_paths, batch_size=batch_size)
    finally:
        embedder.cleanup()


def compute_image_embeddings_with_metadata(
    image_paths: List[Path],
    model_id: str,
    device: str,
    batch_size: int,
    pretrained: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
) -> EmbeddingResult:
    """Compute embeddings and return the successfully embedded path ordering."""
    embedder = ClipEmbedder(
        model_id=model_id,
        device=device,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )
    try:
        return embedder.embed_paths_with_metadata(
            image_paths=image_paths, batch_size=batch_size
        )
    finally:
        embedder.cleanup()
