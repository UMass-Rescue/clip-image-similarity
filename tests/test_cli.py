import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from clip_image_similarity import cli
from clip_image_similarity.config import RunConfig


def make_images(tmp_path: Path, count: int = 3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (2, 2), color=(i, i, i)).save(p)
        paths.append(p.resolve())
    return paths


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
        "compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
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
        "compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
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

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
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


def test_cli_main_entrypoint(monkeypatch, tmp_path):
    """cli.main should parse argv and drive a default pairwise run to completion."""
    images = make_images(tmp_path / "input_main")
    out_dir = tmp_path / "out_main"
    pairwise_path = out_dir / "evaluation_results" / "pairwise_distances.npz"

    monkeypatch.setattr(cli, "find_images", lambda root, exts: images)
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
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
        "compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
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
        "compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0]], dtype=torch.float32
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


def test_cli_main_module_guard(monkeypatch, tmp_path):
    """Executing clip_image_similarity.cli via runpy should still complete a run."""
    images = make_images(tmp_path / "input_guard")
    out_dir = tmp_path / "out_guard"
    # Pre-patch embeddings.compute_image_embeddings so the re-executed module uses the stub.
    monkeypatch.setattr(
        "clip_image_similarity.embeddings.compute_image_embeddings",
        lambda image_paths, model_id, device, batch_size: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        ),
    )
    monkeypatch.setattr(
        cli,
        "compute_image_embeddings",
        lambda *a, **k: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
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
