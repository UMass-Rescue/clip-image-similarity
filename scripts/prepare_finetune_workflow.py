#!/usr/bin/env python3
"""Generate configs and commands for the fine-tune evaluation workflow."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple

import numpy as np
from PIL import Image


MODEL_NAME = "ViT-SO400M-16-SigLIP2-384"
MODEL_ID = "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
MIN_SAMPLES = 5
DISTANCE_THRESHOLD = 0.3
EVAL_K = [1, 5, 10, 20]
SAVE_FREQUENCY = 3
DEFAULT_EPOCHS = 9
DATA_CONFIG_NAME = "data_config.json"
TEST_EVAL_CONFIG_NAME = "test_eval_config.json"
RUN_FINETUNE_NAME = "run_finetune.sh"
COMMANDS_NAME = "commands.txt"
PRECHECKED_LABELS_NAME = "decode_checked_series_labels.json"
DECODE_MANIFEST_NAME = "image_decode_failures.json"
DEDUPED_LABELS_NAME = "deduplicated_series_labels.json"
DEDUP_MANIFEST_NAME = "deduplicated_images.json"
PHASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4
PHASH_IMAGE_SIZE = PHASH_SIZE * PHASH_HIGHFREQ_FACTOR
DEFAULT_PHASH_THRESHOLD = 4
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ImageReference(NamedTuple):
    series: str
    path: str


class ImageHashRecord(NamedTuple):
    path: str
    hash_value: int
    hash_hex: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate workflow configs from a single fine-tune config JSON."
        )
    )
    parser.add_argument(
        "--config",
        default="finetune_config.json",
        help="Path to finetune_config.json.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--deduplicate-images",
        action="store_true",
        help=(
            "Write a de-duplicated labels JSON using 64-bit pHash and point the "
            "generated workflow configs at it."
        ),
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help=(
            "Maximum pHash Hamming distance to treat two images as duplicates "
            f"(default: {DEFAULT_PHASH_THRESHOLD})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        epochs = validate_epochs(args.epochs)
        phash_threshold = validate_phash_threshold(args.phash_threshold)
        config_path = Path(args.config).resolve()
        cfg = load_workflow_config(config_path)
        generated = generate_workflow_files(
            cfg,
            epochs=epochs,
            deduplicate_images=args.deduplicate_images,
            phash_threshold=phash_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Generated workflow files:")
    for path in generated.values():
        print(f"  {path}")
    return 0


def load_workflow_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Fine-tune config must be a JSON object.")

    base_dir = config_path.parent
    dataset_name = require_string(cfg, "dataset_name")
    image_dir = resolve_path(require_string(cfg, "image_dir"), base_dir)
    labels_json = resolve_path(require_string(cfg, "labels_json"), base_dir)
    output_dir = resolve_path(require_string(cfg, "output_dir"), base_dir)

    validate_dataset_name(dataset_name)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir does not exist or is not a directory: {image_dir}")
    if not labels_json.is_file():
        raise FileNotFoundError(f"labels_json does not exist or is not a file: {labels_json}")

    return {
        "dataset_name": dataset_name,
        "image_dir": image_dir,
        "labels_json": labels_json,
        "output_dir": output_dir,
    }


def require_string(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value.strip()


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def validate_dataset_name(dataset_name: str) -> None:
    if not DATASET_NAME_RE.fullmatch(dataset_name):
        raise ValueError(
            "dataset_name must start with a letter or number and contain only "
            "letters, numbers, '.', '_', or '-'."
        )


def validate_epochs(epochs: int) -> int:
    if epochs < SAVE_FREQUENCY or epochs % SAVE_FREQUENCY != 0:
        raise ValueError(
            f"--epochs must be a positive multiple of {SAVE_FREQUENCY}; got {epochs}."
        )
    return epochs


def validate_phash_threshold(threshold: int) -> int:
    if not 0 <= threshold <= PHASH_SIZE * PHASH_SIZE:
        raise ValueError(
            f"--phash-threshold must be between 0 and {PHASH_SIZE * PHASH_SIZE}; "
            f"got {threshold}."
        )
    return threshold


def generate_workflow_files(
    cfg: Dict[str, Any],
    *,
    epochs: int,
    deduplicate_images: bool = False,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> Dict[str, Path]:
    dataset_name = cfg["dataset_name"]
    image_dir = cfg["image_dir"]
    labels_json = cfg["labels_json"]
    output_dir = cfg["output_dir"]

    configs_dir = output_dir / "configs"
    step1_base = output_dir / "step1_processed"
    step2_base = output_dir / "step2_finetune_logs"
    step3_base = output_dir / "step3_eval"
    data_dir = step1_base / MODEL_NAME / dataset_name
    logs_dir = step2_base / MODEL_NAME / dataset_name
    eval_epochs = list(range(SAVE_FREQUENCY, epochs + 1, SAVE_FREQUENCY))

    configs_dir.mkdir(parents=True, exist_ok=True)

    prechecked_labels_path = configs_dir / PRECHECKED_LABELS_NAME
    decode_manifest_path = configs_dir / DECODE_MANIFEST_NAME
    filter_decodable_label_mapping(
        input_json=labels_json,
        output_json=prechecked_labels_path,
        manifest_json=decode_manifest_path,
    )
    labels_json = prechecked_labels_path

    dedupe_manifest_path = configs_dir / DEDUP_MANIFEST_NAME
    if deduplicate_images:
        labels_json = configs_dir / DEDUPED_LABELS_NAME
        deduplicate_label_mapping(
            input_json=prechecked_labels_path,
            output_json=labels_json,
            manifest_json=dedupe_manifest_path,
            phash_threshold=phash_threshold,
        )

    data_config_path = configs_dir / DATA_CONFIG_NAME
    test_eval_config_path = configs_dir / TEST_EVAL_CONFIG_NAME
    run_finetune_path = configs_dir / RUN_FINETUNE_NAME
    commands_path = configs_dir / COMMANDS_NAME

    write_json(
        data_config_path,
        {
            "min_samples": MIN_SAMPLES,
            "output_base_dir": step1_base.as_posix(),
            "models": [{"name": MODEL_NAME, "id": MODEL_ID}],
            "datasets": [
                {
                    "name": dataset_name,
                    "series_image_path": image_dir.as_posix(),
                    "series_labels_path": labels_json.as_posix(),
                    "distance_threshold": DISTANCE_THRESHOLD,
                }
            ],
        },
    )

    write_json(
        test_eval_config_path,
        {
            "dataset": dataset_name,
            "series_image_path": image_dir.as_posix(),
            "step1_base": step1_base.as_posix(),
            "step2_base": step2_base.as_posix(),
            "step3_base": step3_base.as_posix(),
            "series_labels": "data/test_series.json",
            "subseries_labels": "data/test_series.json",
            "eval_k": EVAL_K,
            "eval_epochs": eval_epochs,
            "models": [{"name": MODEL_NAME, "id": MODEL_ID}],
        },
    )

    run_finetune_path.write_text(
        build_run_finetune_script(
            data_dir=data_dir,
            logs_dir=logs_dir,
            dataset_name=dataset_name,
            epochs=epochs,
        ),
        encoding="utf-8",
    )
    run_finetune_path.chmod(0o755)

    commands_path.write_text(
        build_commands_text(
            output_dir=output_dir,
            data_config_path=data_config_path,
            test_eval_config_path=test_eval_config_path,
            run_finetune_path=run_finetune_path,
        ),
        encoding="utf-8",
    )

    return {
        "data_config": data_config_path,
        "test_eval_config": test_eval_config_path,
        "run_finetune": run_finetune_path,
        "commands": commands_path,
        "decode_checked_labels": prechecked_labels_path,
        "decode_manifest": decode_manifest_path,
        **(
            {
                "deduplicated_labels": labels_json,
                "dedupe_manifest": dedupe_manifest_path,
            }
            if deduplicate_images
            else {}
        ),
    }


def filter_decodable_label_mapping(
    *,
    input_json: Path,
    output_json: Path,
    manifest_json: Path,
) -> Dict[str, List[str]]:
    labels = load_label_mapping(input_json)
    ordered_paths: List[str] = []
    affected_series: Dict[str, List[str]] = {}
    for series, paths in labels.items():
        for raw_path in paths:
            path = Path(raw_path).resolve().as_posix()
            if path not in affected_series:
                ordered_paths.append(path)
                affected_series[path] = []
            if series not in affected_series[path]:
                affected_series[path].append(series)

    decodable_paths: set[str] = set()
    failures = []
    warning_records = []
    for path in ordered_paths:
        try:
            image_warnings = decode_image(Path(path))
        except Exception as exc:
            failures.append(
                {
                    "path": path,
                    "affected_series": affected_series[path],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        decodable_paths.add(path)
        if image_warnings:
            warning_records.append(
                {
                    "path": path,
                    "affected_series": affected_series[path],
                    "warnings": image_warnings,
                }
            )

    cleaned: Dict[str, List[str]] = {}
    removed_references = []
    for series, paths in labels.items():
        retained: List[str] = []
        for raw_path in paths:
            path = Path(raw_path).resolve().as_posix()
            if path in decodable_paths:
                retained.append(path)
            else:
                removed_references.append({"series": series, "path": path})
        cleaned[series] = retained

    manifest = {
        "input_series_count": len(labels),
        "input_image_reference_count": sum(len(paths) for paths in labels.values()),
        "input_unique_image_count": len(ordered_paths),
        "output_series_count": len(cleaned),
        "output_image_reference_count": sum(len(paths) for paths in cleaned.values()),
        "failed_unique_image_count": len(failures),
        "removed_image_reference_count": len(removed_references),
        "warning_unique_image_count": len(warning_records),
        "decode_failures": failures,
        "removed_image_references": removed_references,
        "decode_warnings": warning_records,
    }

    write_json(output_json, cleaned)
    write_json(manifest_json, manifest)
    return cleaned


def decode_image(path: Path) -> List[Dict[str, str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(path) as image:
            image.load()
            image.convert("RGB").close()
    return [
        {
            "warning_type": warning.category.__name__,
            "warning_message": str(warning.message),
        }
        for warning in caught
    ]


def deduplicate_label_mapping(
    *,
    input_json: Path,
    output_json: Path,
    manifest_json: Path,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> Dict[str, List[str]]:
    labels = load_label_mapping(input_json)
    image_records = hash_unique_images(iter_image_references(labels))

    kept_by_path: Dict[str, ImageHashRecord] = {}
    duplicates_by_path: Dict[str, Dict[str, Any]] = {}
    hash_errors = []
    for path, result in image_records.items():
        if isinstance(result, Exception):
            hash_errors.append(
                {
                    "path": path,
                    "error_type": type(result).__name__,
                    "error_message": str(result),
                }
            )
            kept_by_path[path] = ImageHashRecord(path=path, hash_value=-1, hash_hex="")
            continue

        duplicate_of = find_duplicate(result, kept_by_path.values(), phash_threshold)
        if duplicate_of is None:
            kept_by_path[path] = result
        else:
            duplicates_by_path[path] = {
                "path": path,
                "hash": result.hash_hex,
                "duplicate_of": duplicate_of.path,
                "duplicate_of_hash": duplicate_of.hash_hex,
                "hamming_distance": hamming_distance(
                    result.hash_value, duplicate_of.hash_value
                ),
            }

    cleaned: Dict[str, List[str]] = {}
    removed_references = []
    globally_seen_paths: set[str] = set()
    for series, paths in labels.items():
        retained: List[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            path = Path(raw_path).resolve().as_posix()
            if path in duplicates_by_path:
                removed_references.append({"series": series, **duplicates_by_path[path]})
                continue
            if path in seen or path in globally_seen_paths:
                removed_references.append(
                    {
                        "series": series,
                        "path": path,
                        "duplicate_of": path,
                        "hamming_distance": 0,
                        "reason": (
                            "repeated within series"
                            if path in seen
                            else "repeated across series"
                        ),
                    }
                )
                continue
            retained.append(path)
            seen.add(path)
            globally_seen_paths.add(path)
        cleaned[series] = retained

    manifest = {
        "hash_algorithm": "phash",
        "hash_bits": PHASH_SIZE * PHASH_SIZE,
        "phash_threshold": phash_threshold,
        "input_series_count": len(labels),
        "input_image_reference_count": sum(len(paths) for paths in labels.values()),
        "input_unique_image_count": len(image_records),
        "output_series_count": len(cleaned),
        "output_image_reference_count": sum(len(paths) for paths in cleaned.values()),
        "removed_unique_image_count": len(duplicates_by_path),
        "removed_image_reference_count": len(removed_references),
        "hash_error_count": len(hash_errors),
        "removed_image_references": removed_references,
        "hash_errors": hash_errors,
    }

    write_json(output_json, cleaned)
    write_json(manifest_json, manifest)
    return cleaned


def load_label_mapping(labels_path: Path) -> Dict[str, List[str]]:
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("labels_json must be a JSON object mapping series to paths.")

    labels: Dict[str, List[str]] = {}
    for series, paths in data.items():
        if not isinstance(series, str):
            raise ValueError("All series names in labels_json must be strings.")
        if not isinstance(paths, list):
            raise ValueError(f"Series '{series}' must map to a list of image paths.")
        labels[series] = []
        for idx, path in enumerate(paths):
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    f"Series '{series}' has a non-string image path at index {idx}."
                )
            labels[series].append(path)
    return labels


def iter_image_references(labels: Dict[str, List[str]]) -> Iterable[ImageReference]:
    for series, paths in labels.items():
        for path in paths:
            yield ImageReference(series=series, path=Path(path).resolve().as_posix())


def hash_unique_images(
    references: Iterable[ImageReference],
) -> Dict[str, ImageHashRecord | Exception]:
    records: Dict[str, ImageHashRecord | Exception] = {}
    for reference in references:
        if reference.path in records:
            continue
        try:
            hash_value = phash_image(Path(reference.path))
        except Exception as exc:
            records[reference.path] = exc
        else:
            records[reference.path] = ImageHashRecord(
                path=reference.path,
                hash_value=hash_value,
                hash_hex=f"{hash_value:016x}",
            )
    return records


def find_duplicate(
    candidate: ImageHashRecord,
    retained: Iterable[ImageHashRecord],
    threshold: int,
) -> ImageHashRecord | None:
    best: tuple[int, ImageHashRecord] | None = None
    for record in retained:
        if record.hash_value < 0:
            continue
        distance = hamming_distance(candidate.hash_value, record.hash_value)
        if distance <= threshold and (best is None or distance < best[0]):
            best = (distance, record)
    return None if best is None else best[1]


def phash_image(path: Path) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Image.open(path) as image:
            pixels = (
                image.convert("L")
                .resize((PHASH_IMAGE_SIZE, PHASH_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            matrix = np.asarray(pixels, dtype=np.float32)

    dct = dct_matrix(PHASH_IMAGE_SIZE) @ matrix @ dct_matrix(PHASH_IMAGE_SIZE).T
    low_freq = dct[:PHASH_SIZE, :PHASH_SIZE]
    median = float(np.median(low_freq.reshape(-1)[1:]))
    bits = low_freq > median

    hash_value = 0
    for bit in bits.reshape(-1):
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


@lru_cache(maxsize=None)
def dct_matrix(size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float32)
    scale0 = np.sqrt(1.0 / size)
    scale = np.sqrt(2.0 / size)
    for k in range(size):
        alpha = scale0 if k == 0 else scale
        for n in range(size):
            matrix[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2 * size))
    return matrix


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def build_run_finetune_script(
    *, data_dir: Path, logs_dir: Path, dataset_name: str, epochs: int
) -> str:
    args = [
        "bash",
        "finetune_workflow.sh",
        "--data-dir",
        data_dir.as_posix(),
        "--logs-dir",
        logs_dir.as_posix(),
        "--dataset-name",
        dataset_name,
        "--model",
        MODEL_ID,
        "--epochs",
        str(epochs),
        "--save-frequency",
        str(SAVE_FREQUENCY),
    ]
    return "#!/usr/bin/env bash\nset -euo pipefail\n\n" + shell_join(args) + "\n"


def build_commands_text(
    *,
    output_dir: Path,
    data_config_path: Path,
    test_eval_config_path: Path,
    run_finetune_path: Path,
) -> str:
    lines = [
        "Run these commands after setting up the two repositories.",
        "",
        "In clip-image-similarity:",
        shell_join(["python", "run_data_workflows.py", "--config", data_config_path.as_posix()]),
        "",
        "In 2026-project-open-clip-sem-hash/src:",
        shell_join(["bash", run_finetune_path.as_posix()]),
        "",
        "Back in clip-image-similarity:",
        shell_join(["python", "run_eval.py", "--config", test_eval_config_path.as_posix()]),
        "",
        "Final outputs:",
        f"{output_dir.as_posix()}/step3_eval",
    ]
    return "\n".join(lines) + "\n"


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


if __name__ == "__main__":
    sys.exit(main())
