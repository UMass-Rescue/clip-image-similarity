import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from clip_image_similarity import cli
from clip_image_similarity.config import RunConfig
from clip_image_similarity.embeddings import EmbeddingResult, SkippedImage


def make_images(tmp_path: Path, count: int = 3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (2, 2), color=(i, i, i)).save(p)
        paths.append(p.resolve())
    return paths


def make_embedding_result(image_paths, rows, skipped=None):
    return EmbeddingResult(
        embeddings=torch.tensor(rows, dtype=torch.float32),
        image_paths=list(image_paths),
        skipped_images=skipped or [],
    )


def assert_benchmark_common(
    out_dir: Path,
    *,
    output_mode: str,
    image_count: int,
    model_id: str = "mock",
    batch_size: int = 1,
    device: str = "cpu",
):
    benchmark = json.loads((out_dir / "benchmark.json").read_text())
    assert benchmark["model_id"] == model_id
    assert benchmark["batch_size"] == batch_size
    assert benchmark["device"] == device
    assert benchmark["image_count"] == image_count
    assert benchmark["embedding_dim"] == 2
    assert benchmark["output_mode"] == output_mode
    assert benchmark["pairwise_output_bytes"] > 0
    assert benchmark["images_per_second_total"] > 0
    assert benchmark["images_per_second_embedding"] > 0
    assert benchmark["cuda_device_name"] is None
    assert benchmark["cuda_peak_memory_allocated_bytes"] is None
    assert benchmark["cuda_peak_memory_reserved_bytes"] is None
    for key in (
        "image_discovery_seconds",
        "embedding_seconds",
        "similarity_seconds",
        "output_write_seconds",
        "total_seconds",
    ):
        assert benchmark[key] >= 0
    return benchmark


def test_cli_run_pairwise(monkeypatch, tmp_path):
    """End-to-end pairwise run should save flattened distances and image paths.

    By stubbing discovery and embeddings we expect cli.run to finish, emit the
    pairwise npz with float32 metadata, and persist the image ordering JSON.
    """
    images = make_images(tmp_path / "input")
    out_dir = tmp_path / "out_pairwise"

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    config = RunConfig(
        input_dir=tmp_path / "input",
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
    )

    cli.run(config)

    pairwise = out_dir / "evaluation_results" / "pairwise_distances.npz"
    assert pairwise.exists()
    data = np.load(pairwise, allow_pickle=False)
    assert "distances" in data and data["dtype"] == "float32"
    paths = json.loads((out_dir / "image_paths.json").read_text())
    assert paths == [p.as_posix() for p in images]
    benchmark = assert_benchmark_common(
        out_dir, output_mode="pairwise", image_count=len(images)
    )
    assert benchmark["pairwise_dtype"] == "float32"
    assert benchmark["top_k"] is None
    assert "label_mapping_seconds" not in benchmark


def test_cli_skips_failed_images_from_outputs(monkeypatch, tmp_path):
    images = make_images(tmp_path / "input_skip")
    out_dir = tmp_path / "out_skip"
    skipped = [
        SkippedImage(
            path=images[1],
            index=1,
            error_type="UnidentifiedImageError",
            error_message="cannot identify image file",
        )
    ]

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            [images[0], images[2]],
            [[1.0, 0.0], [0.0, 1.0]],
            skipped=skipped,
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    cfg = RunConfig(
        input_dir=tmp_path / "input_skip",
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
    )

    cli.run(cfg)

    paths = json.loads((out_dir / "image_paths.json").read_text())
    assert paths == [images[0].as_posix(), images[2].as_posix()]
    data = np.load(
        out_dir / "evaluation_results" / "pairwise_distances.npz",
        allow_pickle=False,
    )
    assert data["distances"].shape == (1,)
    skipped_manifest = json.loads((out_dir / "skipped_images.json").read_text())
    assert skipped_manifest["skipped_image_count"] == 1
    assert skipped_manifest["skipped_images"][0]["path"] == images[1].as_posix()
    benchmark = assert_benchmark_common(
        out_dir, output_mode="pairwise", image_count=2
    )
    assert benchmark["discovered_image_count"] == 3
    assert benchmark["skipped_image_count"] == 1


def test_cli_filters_labels_in_memory_after_skips(monkeypatch, tmp_path):
    images = make_images(tmp_path / "input_label_skip")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "keep": [images[0].as_posix(), images[1].as_posix()],
                "drop": [images[2].as_posix()],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_label_skip"
    skipped = [
        SkippedImage(
            path=images[2],
            index=2,
            error_type="OSError",
            error_message="truncated",
        )
    ]

    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            [images[0], images[1]],
            [[1.0, 0.0], [0.0, 1.0]],
            skipped=skipped,
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    cfg = RunConfig(
        input_dir=tmp_path / "input_label_skip",
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        labels_path=labels_path,
    )

    cli.run(cfg)

    assert json.loads((out_dir / "series_to_indices.json").read_text()) == {
        "keep": [0, 1]
    }
    assert not (out_dir / "loadable_series.json").exists()


def test_cli_all_skipped_images_writes_manifest_and_fails(monkeypatch, tmp_path):
    images = make_images(tmp_path / "input_all_skip", count=1)
    out_dir = tmp_path / "out_all_skip"
    skipped = [
        SkippedImage(
            path=images[0],
            index=0,
            error_type="OSError",
            error_message="cannot load",
        )
    ]

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            [],
            [],
            skipped=skipped,
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    cfg = RunConfig(
        input_dir=tmp_path / "input_all_skip",
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
    )

    with pytest.raises(RuntimeError, match="No images could be loaded"):
        cli.run(cfg)

    assert (out_dir / "skipped_images.json").exists()
    assert not (out_dir / "image_paths.json").exists()


def test_cli_run_pairwise_float16(monkeypatch, tmp_path):
    """Pairwise run with float16 storage should downcast distances on disk.

    With deterministic embeddings we expect the saved npz file to contain
    float16 distances and metadata after the run completes successfully.
    """
    images = make_images(tmp_path / "input_f16")
    out_dir = tmp_path / "out_pairwise_f16"
    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)
    config = RunConfig(
        input_dir=tmp_path / "input_f16",
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        pairwise_dtype="float16",
        overwrite=True,
    )
    cli.run(config)
    data = np.load(
        out_dir / "evaluation_results" / "pairwise_distances.npz", allow_pickle=False
    )
    assert data["distances"].dtype == np.float16
    assert data["dtype"] == "float16"


def test_cli_run_topk_with_labels(monkeypatch, tmp_path):
    """Top-k mode should emit sparse neighbors plus series-to-indices mapping.

    The CLI should save pairwise_topk.npz with the requested dtype/k metadata,
    and since labels_path is provided it must also write series_to_indices.json.
    """
    images = make_images(tmp_path / "input_topk")
    labels_payload = {"series": [p.as_posix() for p in images[:2]]}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels_payload), encoding="utf-8")
    out_dir = tmp_path / "out_topk"

    # find_images is no longer called when labels_path is set; labeled paths are used directly
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0]]
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    config = RunConfig(
        input_dir=tmp_path / "input_topk",
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        top_k=1,
        labels_path=labels_path,
        overwrite=True,
    )

    cli.run(config)

    topk_path = out_dir / "evaluation_results" / "pairwise_topk.npz"
    assert topk_path.exists()
    data = np.load(topk_path, allow_pickle=False)
    assert data["dtype"] == "float32"
    assert data["top_k"] == 1
    series_indices = json.loads((out_dir / "series_to_indices.json").read_text())
    assert series_indices == {"series": [0, 1]}
    # Only the 2 labeled images are embedded — the 3rd image in input_topk is excluded
    benchmark = assert_benchmark_common(
        out_dir, output_mode="top_k", image_count=2
    )
    assert benchmark["top_k"] == 1
    assert benchmark["label_mapping_seconds"] >= 0


def test_cli_main_entrypoint(monkeypatch, tmp_path):
    """cli.main should parse argv and drive a default pairwise run to completion."""
    images = make_images(tmp_path / "input_main")
    out_dir = tmp_path / "out_main"
    pairwise_path = out_dir / "evaluation_results" / "pairwise_distances.npz"

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    argv = [
        "clip-pairwise-eval",
        "--input-dir",
        str(tmp_path / "input_main"),
        "--output-dir",
        str(out_dir),
        "--model",
        "mock",
        "--batch-size",
        "1",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert pairwise_path.exists()


def test_cli_existing_topk_and_pairwise_files(monkeypatch, tmp_path):
    """Running without --overwrite should raise when outputs already exist.

    We pre-create both pairwise and top-k artifacts; cli.run must surface
    FileExistsError to keep users from clobbering results accidentally.
    """
    images = make_images(tmp_path / "input_exist")
    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    # Pairwise exists
    out_pairwise = tmp_path / "out_exists"
    eval_dir = out_pairwise / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "pairwise_distances.npz").write_bytes(b"data")
    original_exists = Path.exists

    def fake_exists(self):
        if self == out_pairwise:
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    cfg_pairwise = RunConfig(
        input_dir=tmp_path / "input_exist",
        output_dir=out_pairwise,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        overwrite=False,
    )
    with pytest.raises(FileExistsError):
        cli.run(cfg_pairwise)

    # Top-k exists
    out_topk = tmp_path / "out_exists_topk"
    eval_dir_topk = out_topk / "evaluation_results"
    eval_dir_topk.mkdir(parents=True, exist_ok=True)
    (eval_dir_topk / "pairwise_topk.npz").write_bytes(b"data")
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: False if self == out_topk else original_exists(self),
    )
    cfg_topk = RunConfig(
        input_dir=tmp_path / "input_exist",
        output_dir=out_topk,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        top_k=1,
        overwrite=False,
    )
    with pytest.raises(FileExistsError):
        cli.run(cfg_topk)


def test_cli_errors(monkeypatch, tmp_path):
    """Bundle of negative scenarios to ensure cli.run enforces its guards.

    Missing labels should trigger FileNotFoundError, pre-existing output dirs
    should block unless overwrite is set, empty discovery raises RuntimeError,
    top-k >= num_images raises ValueError, and existing eval files should stop
    the run to prevent silent overwrites.
    """
    input_dir = tmp_path / "input_err"
    input_dir.mkdir()
    out_dir = tmp_path / "out_err"
    # missing labels path
    cfg = RunConfig(
        input_dir=input_dir,
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        labels_path=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError):
        cli.run(cfg)

    cfg_missing_checkpoint = RunConfig(
        input_dir=input_dir,
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        checkpoint_path=tmp_path / "missing.pt",
    )
    with pytest.raises(FileNotFoundError):
        cli.run(cfg_missing_checkpoint)

    out_dir.mkdir()
    cfg_no_overwrite = RunConfig(
        input_dir=input_dir,
        output_dir=out_dir,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
    )
    with pytest.raises(FileExistsError):
        cli.run(cfg_no_overwrite)

    monkeypatch.setattr(cli, "find_images", lambda root, exts: [])
    cfg_no_images = RunConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "out_err2",
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
    )
    with pytest.raises(RuntimeError):
        cli.run(cfg_no_images)

    monkeypatch.setattr(cli, "find_images", lambda root, exts: [input_dir / "a.png"])
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0]]
        ),
    )
    cfg_topk_large = RunConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "out_err3",
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
        top_k=1,
    )
    with pytest.raises(ValueError):
        cli.run(cfg_topk_large)

    monkeypatch.setattr(cli, "find_images", lambda root, exts: [input_dir / "a.png"])
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0]]
        ),
    )
    # Create existing pairwise file to trigger overwrite protection
    out_exist = tmp_path / "out_err4"
    eval_dir = out_exist / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "pairwise_distances.npz").write_bytes(b"data")
    cfg_exists = RunConfig(
        input_dir=input_dir,
        output_dir=out_exist,
        model_id="mock",
        batch_size=1,
        device="cpu",
        image_exts=(".png",),
    )
    with pytest.raises(FileExistsError):
        cli.run(cfg_exists)


def test_parse_args_defaults(monkeypatch, tmp_path):
    """Blank --image-exts input should fall back to DEFAULT_EXTS in config."""
    input_dir = tmp_path / "input_parse"
    output_dir = tmp_path / "output_parse"
    input_dir.mkdir()
    argv = [
        "prog",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--image-exts",
        "",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = cli.parse_args_to_config()
    assert cfg.image_exts == cli.DEFAULT_EXTS


def test_parse_args_loader_options(monkeypatch, tmp_path):
    input_dir = tmp_path / "input_parse_loader"
    output_dir = tmp_path / "output_parse_loader"
    checkpoint_path = tmp_path / "epoch_4.pt"
    input_dir.mkdir()
    checkpoint_path.touch()
    argv = [
        "prog",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        "ViT-H-14-378-quickgelu",
        "--pretrained",
        "dfn5b",
        "--checkpoint_path",
        str(checkpoint_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cfg = cli.parse_args_to_config()

    assert cfg.model_id == "ViT-H-14-378-quickgelu"
    assert cfg.pretrained == "dfn5b"
    assert cfg.checkpoint_path == checkpoint_path.resolve()


def test_cli_run_forwards_loader_options(monkeypatch, tmp_path):
    images = make_images(tmp_path / "input_loader")
    checkpoint_path = tmp_path / "epoch_4.pt"
    checkpoint_path.touch()
    captured = {}

    def fake_compute_image_embeddings_with_metadata(
        image_paths,
        model_id,
        device,
        batch_size,
        pretrained=None,
        checkpoint_path=None,
    ):
        captured.update(
            image_paths=image_paths,
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
        return make_embedding_result(image_paths[:2], [[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        fake_compute_image_embeddings_with_metadata,
    )
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)

    cfg = RunConfig(
        input_dir=tmp_path / "input_loader",
        output_dir=tmp_path / "out_loader",
        model_id="ViT-H-14-378-quickgelu",
        batch_size=2,
        device="cpu",
        image_exts=(".png",),
        pretrained="dfn5b",
        checkpoint_path=checkpoint_path,
        overwrite=True,
    )

    cli.run(cfg)

    assert captured["model_id"] == "ViT-H-14-378-quickgelu"
    assert captured["pretrained"] == "dfn5b"
    assert captured["checkpoint_path"] == checkpoint_path.resolve()
    assert captured["device"] == "cpu"
    assert captured["batch_size"] == 2
    assert captured["image_paths"] == images


def test_cli_main_module_guard(monkeypatch, tmp_path):
    """Executing clip_image_similarity.cli via runpy should still complete a run."""
    images = make_images(tmp_path / "input_guard")
    out_dir = tmp_path / "out_guard"
    # Pre-patch embeddings.compute_image_embeddings_with_metadata so the re-executed module uses the stub.
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.compute_image_embeddings_with_metadata",
        lambda image_paths, model_id, device, batch_size: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    )
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings_with_metadata",
        lambda image_paths, *a, **k: make_embedding_result(
            image_paths, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    )
    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(cli, "configure_logging", lambda log_file: None)
    argv = [
        "clip_image_similarity.cli",
        "--input-dir",
        str(tmp_path / "input_guard"),
        "--output-dir",
        str(out_dir),
        "--model",
        "mock",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    import runpy

    runpy.run_module("clip_image_similarity.cli", run_name="__main__")
    assert (out_dir / "evaluation_results" / "pairwise_distances.npz").exists()
