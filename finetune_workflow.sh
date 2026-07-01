#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=
MODEL=
DATASET_NAME=
BATCH_SIZE=8
SERIES_EMBEDDING_BATCH_SIZE=128
LR=1e-5
WD=0.1
EPOCHS=30
WARMUP=100
WORKERS=4
LOGS_DIR=
DEVICE=cuda
SAVE_FREQUENCY=3
REPORT_TO=tensorboard
SIGLIP=1
LR_SCHEDULE_PREFIX_EPOCHS=
LR_EXTENSION_LR=

usage() {
  cat <<'EOF'
Usage:
  bash finetune_workflow.sh \
    --data-dir /path/to/step1_processed/<model_name>/<dataset_name> \
    --logs-dir /path/to/step2_finetune_logs/<model_name>/<dataset_name> \
    --dataset-name <dataset_name> \
    --model <open_clip_model_id>

Required:
  --data-dir      Processed data directory containing data/ and subseries_split_data/.
  --logs-dir      Directory where open_clip_train writes series/subseries run folders.
  --dataset-name  Dataset/run prefix. Checkpoints are written under
                  <logs-dir>/<dataset-name>-series/checkpoints/ and
                  <logs-dir>/<dataset-name>-subseries/checkpoints/.
  --model         OpenCLIP model id, e.g. hf-hub:apple/DFN5B-CLIP-ViT-H-14-378.

Optional:
  --batch-size <int>                    Default: 8
  --series-embedding-batch-size <int>   Default: 128
  --lr <float>                          Default: 1e-5
  --wd <float>                          Default: 0.1
  --epochs <int>                        Default: 30
  --warmup <int>                        Default: 100
  --workers <int>                       Default: 4
  --device <device>                     Default: cuda
  --save-frequency <int>                Default: 3
  --lr-schedule-prefix-epochs <int>     Preserve a shorter cosine LR schedule through this epoch.
  --lr-extension-lr <float>             Restart LR for epochs after --lr-schedule-prefix-epochs.
  --report-to <target>                  Default: tensorboard
  --no-siglip                           Do not pass --siglip to open_clip_train.
  -h, --help                            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-siglip|-h|--help) ;;
    --*)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac

  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --series-embedding-batch-size) SERIES_EMBEDDING_BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --wd) WD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --logs-dir) LOGS_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --save-frequency) SAVE_FREQUENCY="$2"; shift 2 ;;
    --lr-schedule-prefix-epochs) LR_SCHEDULE_PREFIX_EPOCHS="$2"; shift 2 ;;
    --lr-extension-lr) LR_EXTENSION_LR="$2"; shift 2 ;;
    --report-to) REPORT_TO="$2"; shift 2 ;;
    --no-siglip) SIGLIP=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

require_arg() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing required argument: $name" >&2
    usage >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_arg "--data-dir" "$DATA_DIR"
require_arg "--logs-dir" "$LOGS_DIR"
require_arg "--dataset-name" "$DATASET_NAME"
require_arg "--model" "$MODEL"

SERIES_TRAIN_DATA="$DATA_DIR/data/train_series.json"
SERIES_VAL_DATA="$DATA_DIR/data/val_series.json"
SUBSERIES_TRAIN_DATA="$DATA_DIR/subseries_split_data/train_series.json"
SUBSERIES_VAL_DATA="$DATA_DIR/subseries_split_data/val_series.json"
SERIES_RUN_NAME="${DATASET_NAME}-series"
SUBSERIES_RUN_NAME="${DATASET_NAME}-subseries"

require_file "$SERIES_TRAIN_DATA"
require_file "$SERIES_VAL_DATA"
require_file "$SUBSERIES_TRAIN_DATA"
require_file "$SUBSERIES_VAL_DATA"
mkdir -p "$LOGS_DIR"

COMMON_ARGS=(
  --save-frequency "$SAVE_FREQUENCY"
  --zeroshot-frequency 0
  --dataset-type series_json
  --model "$MODEL"
  --batch-size "$BATCH_SIZE"
  --series-embedding-batch-size "$SERIES_EMBEDDING_BATCH_SIZE"
  --lr "$LR"
  --wd "$WD"
  --epochs "$EPOCHS"
  --warmup "$WARMUP"
  --workers "$WORKERS"
  --device "$DEVICE"
  --logs "$LOGS_DIR"
  --report-to "$REPORT_TO"
)

if [[ "$SIGLIP" == "1" ]]; then
  COMMON_ARGS+=(--siglip)
fi

if [[ -n "$LR_SCHEDULE_PREFIX_EPOCHS" ]]; then
  COMMON_ARGS+=(--lr-schedule-prefix-epochs "$LR_SCHEDULE_PREFIX_EPOCHS")
fi

if [[ -n "$LR_EXTENSION_LR" ]]; then
  COMMON_ARGS+=(--lr-extension-lr "$LR_EXTENSION_LR")
fi

echo -e "\n=== Finetune on series ==="
python -m open_clip_train.main \
  "${COMMON_ARGS[@]}" \
  --train-data "$SERIES_TRAIN_DATA" \
  --val-data "$SERIES_VAL_DATA" \
  --name "$SERIES_RUN_NAME"

echo -e "\n=== Finetune on subseries ==="
python -m open_clip_train.main \
  "${COMMON_ARGS[@]}" \
  --train-data "$SUBSERIES_TRAIN_DATA" \
  --val-data "$SUBSERIES_VAL_DATA" \
  --name "$SUBSERIES_RUN_NAME"
