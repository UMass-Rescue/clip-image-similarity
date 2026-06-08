SERIES_IMAGE_PATH=/home/prasanna/Downloads/CUHK_03/cuhk_labeled/images
SERIES_LABELS_PATH=/home/prasanna/Downloads/CUHK_03/cuhk_labeled/flat_labels.json
OUTPUT_BASE_DIR=./out/cuhk_end_to_end
MODEL=hf-hub:timm/ViT-SO400M-14-SigLIP2-378
MIN_SAMPLES=5
DISTANCE_THRESHOLD=0.3

while [[ $# -gt 0 ]]; do
  case $1 in
    --series-image-path)    SERIES_IMAGE_PATH="$2";  shift 2 ;;
    --series-labels-path)   SERIES_LABELS_PATH="$2"; shift 2 ;;
    --output-base-dir)      OUTPUT_BASE_DIR="$2";    shift 2 ;;
    --model)                MODEL="$2";              shift 2 ;;
    --min-samples)          MIN_SAMPLES="$2";        shift 2 ;;
    --distance-threshold)   DISTANCE_THRESHOLD="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

source .venv/bin/activate

LOADABLE_LABELS_PATH="$OUTPUT_BASE_DIR/data/loadable_series.json"
SKIPPED_IMAGES_PATH="$OUTPUT_BASE_DIR/data/skipped_images.json"

echo -e "\n=== Step 0: Filter labels to loadable images ==="
python -m scripts.filter_loadable_labels \
  --labels-json "$SERIES_LABELS_PATH" \
  --output-json "$LOADABLE_LABELS_PATH" \
  --skipped-json "$SKIPPED_IMAGES_PATH" \
  --min-images-per-series 2

echo -e "\n=== Step 1: Split dataset into train/val/test ==="
python -m scripts.split_series_json \
  --series-json "$LOADABLE_LABELS_PATH" \
  --out-dir "$OUTPUT_BASE_DIR/data" \
  --train-pct 70 \
  --val-pct 15 \
  --test-pct 15 \
  --seed 42 \
  --combine-train-val

echo -e "\n=== Step 2: Compute pairwise distances (trainval) ==="
make run \
  INPUT_DIR="$SERIES_IMAGE_PATH" \
  MODEL="$MODEL" \
  BATCH_SIZE=200 \
  DEVICE=cuda \
  PAIRWISE_DTYPE=float16 \
  ANONYMIZE_LABELS="$OUTPUT_BASE_DIR/data/trainval_series.json" \
  OUTPUT_DIR="$OUTPUT_BASE_DIR/distances"

echo -e "\n=== Step 3: Generate distance histogram ==="
python -m metrics.distance_histogram \
  --pairwise-output-dir "$OUTPUT_BASE_DIR/distances" \
  --bins 200 \
  --range-max 1.0 \
  --density \
  --labeled-images-only

if [ -t 0 ]; then
  read -rp "Distance threshold [default: $DISTANCE_THRESHOLD]: " input
  DISTANCE_THRESHOLD="${input:-$DISTANCE_THRESHOLD}"
fi
CLUSTER_DIR="$OUTPUT_BASE_DIR/distances/clustered_series_by_distance_t${DISTANCE_THRESHOLD//./_}_average_min$MIN_SAMPLES"

echo -e "\n=== Step 4: Cluster trainval into subseries ==="
python -m scripts.cluster_series_by_distance \
    --pairwise-output-dir "$OUTPUT_BASE_DIR/distances" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --min-samples "$MIN_SAMPLES"

echo -e "\n=== Step 5: Split subseries into train/val ==="
python -m scripts.split_series_json \
    --series-json "$CLUSTER_DIR/series_to_image_paths.json" \
    --out-dir "$OUTPUT_BASE_DIR/subseries_split_data" \
    --train-pct 85 \
    --val-pct 15
