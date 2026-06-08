#!/usr/bin/env python3
"""Generate configs and commands for the fine-tune evaluation workflow."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict


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
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        epochs = validate_epochs(args.epochs)
        config_path = Path(args.config).resolve()
        cfg = load_workflow_config(config_path)
        generated = generate_workflow_files(cfg, epochs=epochs)
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


def generate_workflow_files(cfg: Dict[str, Any], *, epochs: int) -> Dict[str, Path]:
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
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
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
