SERIES_IMAGE_PATH=/home/prasanna/Downloads/CUHK_03/cuhk_labeled/images
SERIES_LABELS_PATH=/home/prasanna/Downloads/CUHK_03/cuhk_labeled/flat_labels.json
OUTPUT_BASE_DIR=./out/cuhk_end_to_end
MODEL=hf-hub:timm/ViT-SO400M-14-SigLIP2-378
MIN_SAMPLES=5
DISTANCE_THRESHOLD=0.3

source .venv/bin/activate

echo -e "\n=== Step 1: Split dataset into train/val/test ==="
python -m scripts.split_series_json \
  --series-json "$SERIES_LABELS_PATH" \
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

read -rp "Distance threshold [default: $DISTANCE_THRESHOLD]: " input
DISTANCE_THRESHOLD="${input:-$DISTANCE_THRESHOLD}"
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
