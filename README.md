# Clip Image Similarity

CLI and library to scan a folder of images, embed them with a CLIP model, and produce a full pairwise distance list using anonymous IDs so raw file paths stay private.

## Quickstart

```bash
python -m clip_image_similarity.cli \
  --input-dir /path/to/images \
  --output-dir /tmp/clip-results \
  --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  --batch-size 16 \
  --device cuda
```

## How to Run

Option 1: Use the Makefile (auto-creates venv and installs deps)
```bash
make run \
  INPUT_DIR=/path/to/images \
  OUTPUT_DIR=/tmp/clip-results \
  MODEL=hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  BATCH_SIZE=16 \
  DEVICE=cuda \
  OVERWRITE=1
```

Example (CPU) mirroring a full run:
```bash
make run INPUT_DIR=./resources/images OUTPUT_DIR=./results MODEL=hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 BATCH_SIZE=32 DEVICE=cpu OVERWRITE=1
```

Option 2: Create venv with Makefile, then run Python manually
```bash
make install                     # sets up .venv and installs requirements
source .venv/bin/activate
python -m clip_image_similarity.cli \
  --input-dir /path/to/images \
  --output-dir /tmp/clip-results \
  --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  --batch-size 16 \
  --device cuda
```

Outputs (parquet by default):
- `file_to_id_map/anon_id_map.json`: absolute path to anonymous numeric ID map (**DO NOT SHARE THIS FILE IF YOU NEED TO KEEP FILE NAMES PRIVATE**).
- `evaluation_results/pairwise_clip_compare.parquet`: all unordered pairwise distances `(1 - cosine_similarity)` using only anonymous IDs (columnar, compressed if you pass `--parquet-compression`).
- `config.json`: run configuration snapshot

If you need JSON instead (less efficient, larger memory use), add `--pairwise-format json` and the output will be `evaluation_results/pairwise_clip_compare.json`.

Progress bars and timestamped logs show progress through discovery, embedding, and distance calculation.

## Output Details and Privacy
- `file_to_id_map/anon_id_map.json`: Maps absolute file paths to anonymous IDs. **Do not share this file** if you need to keep file names private.
- `evaluation_results/pairwise_clip_compare.json`: Pairwise distances using only anonymous IDs. **This file alone is sufficient to share results without revealing paths.**
- `config.json`: Snapshot of the run configuration (paths, model id, etc.).

## Compute mAP (optional)
After generating results, you can compute Mean Average Precision from labels and the pairwise distances (use JSON output from the main run via `--pairwise-format json` or convert the Parquet to JSON):
```bash
python metrics/map.py \
  --labels ./labels.json \
  --pairwise ./results/evaluation_results/pairwise_clip_compare.json \
  --output_csv ./results/metrics/map.csv
```

### Optional: Generate anonymous labels then compute mAP
1) Convert labeled paths to anon IDs (keeps file names private in the labels JSON):
```bash
python label_helpers/labels_to_anon_ids.py \
  --labels ./resources/labels/images_series_labels.json \
  --anon-map ./results/file_to_id_map/anon_id_map.json \
  --output ./results/anon_labels.json
```
2) Run mAP using the anon labels and pairwise distances (evaluates retrieval quality on anonymous IDs):
```bash
python metrics/map.py \
  --labels ./results/anon_labels.json \
  --pairwise ./results/evaluation_results/pairwise_clip_compare.json \
  --output_csv ./results/map.csv
```
