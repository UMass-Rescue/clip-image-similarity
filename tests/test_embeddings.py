from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from clip_image_similarity.embeddings import ClipEmbedder, compute_image_embeddings


class DummyModel:
    def __init__(self):
        self.moved_to = None

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return batch * 2

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self


def dummy_preprocess(img) -> torch.Tensor:
    # Ignore image contents; return a simple tensor
    return torch.tensor([1.0], dtype=torch.float32)


def create_images(tmp_path: Path, count: int):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (2, 2), color=(i, i, i)).save(p)
        paths.append(p)
    return paths


def test_compute_image_embeddings_uses_mocked_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.open_clip.create_model_from_pretrained",
        lambda model_id: (DummyModel(), dummy_preprocess),
    )
    image_paths = create_images(tmp_path, 2)
    embs = compute_image_embeddings(
        image_paths=image_paths,
        model_id="mock",
        device="cpu",
        batch_size=1,
    )
    assert embs.shape == (2, 1)
    # preprocess returns 1.0, model multiplies by 2
    assert torch.allclose(embs, torch.tensor([[2.0], [2.0]], dtype=torch.float32))


def test_clip_embedder_cleanup_calls_cuda_empty_cache(monkeypatch):
    called = {"empty_cache": False}
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.open_clip.create_model_from_pretrained",
        lambda model_id: (DummyModel(), dummy_preprocess),
    )
    monkeypatch.setattr(
        "torch.cuda.empty_cache", lambda: called.update(empty_cache=True)
    )
    embedder = ClipEmbedder(model_id="mock", device="cpu")
    embedder.device = "cuda:0"
    embedder.cleanup()
    assert called["empty_cache"] is True
