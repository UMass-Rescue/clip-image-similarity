from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import RunConfig
from .embeddings import compute_image_embeddings
from .path_id_store import build_id_map_for_paths, create_path_id_store
from .serialization import save_config, save_json
from .similarity import SimilarityComputer
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
        "--pairwise-format",
        choices=["parquet", "json"],
        default="parquet",
        help="Output format for pairwise distances (default: parquet).",
    )
    parser.add_argument(
        "--pairwise-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Numeric precision used when storing pairwise distances (default: float32).",
    )
    parser.add_argument(
        "--pairwise-chunk-size-pairs",
        type=int,
        default=5_000_000,
        help="Number of pairs per chunk when streaming pairwise output (parquet).",
    )
    parser.add_argument(
        "--parquet-compression",
        default=None,
        help="Parquet compression codec (e.g., zstd, gzip). Use none for no compression (default).",
    )
    parser.add_argument(
        "--parquet-compression-level",
        type=int,
        default=None,
        help="Compression level for the selected Parquet codec (if supported).",
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

    return RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_id=args.model,
        batch_size=args.batch_size,
        device=device,
        image_exts=image_exts,
        pairwise_format=args.pairwise_format,
        pairwise_dtype=args.pairwise_dtype,
        pairwise_chunk_size_pairs=args.pairwise_chunk_size_pairs,
        pairwise_compression=None if not args.parquet_compression or args.parquet_compression.lower() == "none" else args.parquet_compression,
        pairwise_compression_level=args.parquet_compression_level,
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

    log("Assigning anonymous IDs to image paths...")
    id_store = create_path_id_store(config.output_dir)
    id_map = build_id_map_for_paths(id_store, image_paths)
    id_store.save()
    log("Anonymous ID map saved.")

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

    ids_in_order = [id_map[str(p)] for p in image_paths]
    eval_dir = config.output_dir / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if config.pairwise_format == "parquet":
        pairwise_path = eval_dir / "pairwise_clip_compare.parquet"
        if pairwise_path.exists() and not config.overwrite:
            raise FileExistsError(f"{pairwise_path} already exists. Use --overwrite to replace it.")
        log(
            f"Streaming pairwise distances to Parquet at {pairwise_path} "
            f"(write_dtype={config.pairwise_dtype}, compression={config.pairwise_compression or 'none'}, "
            f"chunk_size_pairs={config.pairwise_chunk_size_pairs})."
        )
        computer.stream_upper_triangle_to_parquet(
            dist_matrix=dist,
            ids=ids_in_order,
            output_path=pairwise_path,
            chunk_size_pairs=config.pairwise_chunk_size_pairs,
            compression=config.pairwise_compression,
            compression_level=config.pairwise_compression_level,
            write_dtype=torch.float16 if config.pairwise_dtype == "float16" else torch.float32,
        )
    else:
        pairwise_path = eval_dir / "pairwise_clip_compare.json"
        if pairwise_path.exists() and not config.overwrite:
            raise FileExistsError(f"{pairwise_path} already exists. Use --overwrite to replace it.")
        log("Collecting pairwise distances to JSON (may be memory-intensive for large datasets).")
        results = computer.pairwise_distances(image_paths, id_map, dist)
        save_json(results, pairwise_path)

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
