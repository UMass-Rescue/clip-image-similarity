from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import RunConfig
from .embeddings import compute_image_embeddings
from .serialization import save_config, save_json
from .similarity import SimilarityComputer
from .packed_distances import flatten_upper_triangle
from .topk import extract_topk_neighbors, save_topk_neighbors
from .labels import map_labels_to_indices
from .utils import DEFAULT_EXTS, configure_logging, default_device, find_images, log, plural


def parse_args_to_config() -> RunConfig:
    """Parse CLI arguments into a validated RunConfig.

    Returns:
        RunConfig populated from CLI flags with defaults for device, batch size, and extensions.
    """
    parser = argparse.ArgumentParser(description="Compute pairwise CLIP distances for images in a folder.")
    parser.add_argument("--input-dir", "-i", required=True, help="Root directory containing images.")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory where results will be written.")
    parser.add_argument(
        "--model",
        "-m",
        default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
        help="Hugging Face Hub model id for OpenCLIP (e.g. hf-hub:apple/DFN5B-CLIP-ViT-H-14-384).",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for embedding computation.")
    parser.add_argument("--device", "-d", default=None, help="Device to run on (e.g. cuda, cuda:0, cpu). Defaults to CUDA if available.")
    parser.add_argument(
        "--image-exts",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated list of image extensions to include (defaults to common formats).",
    )
    parser.add_argument(
        "--pairwise-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Numeric precision used when storing pairwise distances (default: float32).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k neighbors to store per image instead of full flattened distances.",
    )
    parser.add_argument(
        "--anonymize-labels",
        default=None,
        help="Optional labels JSON (series -> list of image paths); will be converted to series -> list of indices.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.image_exts:
        raw_exts = [e.strip() for e in args.image_exts.split(",") if e.strip()]
        image_exts = tuple(e if e.startswith(".") else f".{e}" for e in raw_exts)
    else:
        image_exts = DEFAULT_EXTS

    device = args.device or default_device()

    labels_path = Path(args.anonymize_labels).resolve() if args.anonymize_labels else None

    return RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_id=args.model,
        batch_size=args.batch_size,
        device=device,
        image_exts=image_exts,
        pairwise_dtype=args.pairwise_dtype,
        top_k=args.top_k,
        labels_path=labels_path,
        overwrite=args.overwrite,
    )


def run(config: RunConfig) -> None:
    """Execute the full pipeline from discovery through saving results.

    Args:
        config: RunConfig instance describing inputs, outputs, and model settings.
    """
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output directory {config.output_dir} already exists. Use --overwrite to replace existing results."
        )
    config.ensure_output_dir()
    configure_logging(config.output_dir / "run.log")
    log("Starting pairwise CLIP evaluation run...")
    log(f"Using model '{config.model_id}' on device '{config.device}'.")
    log(f"Writing outputs under {config.output_dir}.", allow_file=False)

    log(f"Searching for images under {config.input_dir} with extensions {config.image_exts}...", allow_file=False)
    image_paths = find_images(config.input_dir, config.image_exts)
    if not image_paths:
        raise RuntimeError(f"No images found in {config.input_dir} with extensions {config.image_exts}")
    log(f"Found {plural(len(image_paths), 'image')} to process.")

    embeddings = compute_image_embeddings(
        image_paths=image_paths,
        model_id=config.model_id,
        device=config.device,
        batch_size=config.batch_size,
    )

    computer = SimilarityComputer(device=config.device)
    sim = computer.cosine_similarity_matrix(embeddings, dtype=torch.float32)
    dist = computer.similarity_to_distance(sim)
    if config.pairwise_dtype == "float16":
        dist = dist.to(torch.float16)

    eval_dir = config.output_dir / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if config.top_k:
        pairwise_path = eval_dir / "pairwise_topk.npz"
        if pairwise_path.exists() and not config.overwrite:
            raise FileExistsError(f"{pairwise_path} already exists. Use --overwrite to replace it.")
        log(f"Extracting top-{config.top_k} neighbors per image and saving to {pairwise_path}.")
        indices, distances = extract_topk_neighbors(dist, top_k=config.top_k, dtype=config.pairwise_dtype)
        save_topk_neighbors(pairwise_path, indices=indices, distances=distances, dtype=config.pairwise_dtype)
    else:
        pairwise_path = eval_dir / "pairwise_distances.npz"
        if pairwise_path.exists() and not config.overwrite:
            raise FileExistsError(f"{pairwise_path} already exists. Use --overwrite to replace it.")
        log(f"Flattening and saving pairwise distances to {pairwise_path} (dtype={config.pairwise_dtype}).")
        flat_np = flatten_upper_triangle(dist).cpu().numpy()
        np_dtype = np.float16 if config.pairwise_dtype == "float16" else np.float32
        flat_np = flat_np.astype(np_dtype, copy=False)
        np.savez_compressed(pairwise_path, distances=flat_np, dtype=config.pairwise_dtype)

    paths_json = config.output_dir / "image_paths.json"
    save_json([p.as_posix() for p in image_paths], paths_json)
    log(f"Saved image path ordering to {paths_json} (do not share if paths are sensitive).")

    if config.labels_path:
        log("Mapping provided labels to indices...")
        series_indices = map_labels_to_indices(config.labels_path, image_paths)
        series_out = config.output_dir / "series_to_indices.json"
        save_json(series_indices, series_out)
        log(f"Saved series->indices to {series_out} for downstream mAP.")

    save_config(config, config.output_dir)

    log("Run complete.")
    log(f"Saved pairwise distances to {pairwise_path}", allow_file=False)
    log(f"Processed {plural(len(image_paths), 'image')} using model {config.model_id} on {config.device}.")


def main() -> None:
    """Entry point for `python -m clip_image_similarity.cli`; parse args and run the pipeline."""
    config = parse_args_to_config()
    run(config)


if __name__ == "__main__":
    main()
