from pathlib import Path

import numpy as np
import torch
from PIL import Image

from clip_image_similarity.embeddings import (
    ClipEmbedder,
    compute_image_embeddings,
    compute_image_embeddings_with_metadata,
)


class DummyModel:
    def __init__(self):
        self.moved_to = None
        self.eval_called = False

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return batch * 2

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
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


def test_compute_image_embeddings_with_metadata_skips_bad_images(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.open_clip.create_model_from_pretrained",
        lambda model_id: (DummyModel(), dummy_preprocess),
    )
    image_paths = create_images(tmp_path, 2)
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    paths = [image_paths[0], bad, image_paths[1]]

    result = compute_image_embeddings_with_metadata(
        image_paths=paths,
        model_id="mock",
        device="cpu",
        batch_size=2,
    )

    assert result.image_paths == [image_paths[0], image_paths[1]]
    assert result.embeddings.shape == (2, 1)
    assert len(result.skipped_images) == 1
    assert result.skipped_images[0].path == bad
    assert result.skipped_images[0].index == 1


def test_compute_image_embeddings_with_metadata_all_bad_images(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.open_clip.create_model_from_pretrained",
        lambda model_id: (DummyModel(), dummy_preprocess),
    )
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")

    result = compute_image_embeddings_with_metadata(
        image_paths=[bad],
        model_id="mock",
        device="cpu",
        batch_size=1,
    )

    assert result.image_paths == []
    assert result.embeddings.shape == (0, 0)
    assert len(result.skipped_images) == 1


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


def test_clip_embedder_uses_pretrained_and_checkpoint(monkeypatch, tmp_path):
    calls = {"create": None, "checkpoint": None}

    def fake_create_model_from_pretrained(*args, **kwargs):
        calls["create"] = (args, kwargs)
        return DummyModel(), dummy_preprocess

    def fake_load_checkpoint(model, checkpoint_path):
        calls["checkpoint"] = checkpoint_path

    monkeypatch.setattr(
        "clip_image_similarity.embeddings.open_clip.create_model_from_pretrained",
        fake_create_model_from_pretrained,
    )
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.open_clip.load_checkpoint",
        fake_load_checkpoint,
    )

    checkpoint_path = tmp_path / "epoch_4.pt"
    embedder = ClipEmbedder(
        model_id="mock",
        device="cpu",
        pretrained="dfn5b",
        checkpoint_path=checkpoint_path,
    )

    assert calls["create"] == (
        ("mock",),
        {"pretrained": "dfn5b", "device": "cpu"},
    )
    assert calls["checkpoint"] == str(checkpoint_path)
    assert embedder.model.moved_to == "cpu"
    assert embedder.model.eval_called is True
