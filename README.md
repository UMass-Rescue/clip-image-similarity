# Clip Image Similarity

CLI and library to embed a folder of images with a CLIP model and export a compact, flattened pairwise distance array for series retrieval evaluation.

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

## Outputs
- `evaluation_results/pairwise_distances.npz`: flattened upper-triangular distances `(1 - cosine_similarity)`; dtype `float32` (default) or `float16` via `--pairwise-dtype`.
- `image_paths.json`: ordered list of image paths corresponding to indices in the flattened array. **DO NOT SHARE THIS FILE IF YOU NEED TO KEEP FILE NAMES PRIVATE**
- `series_to_indices.json` (optional): only written if `--anonymize-labels` is provided; maps series -> list of indices for downstream mAP without exposing paths.
- `config.json`: run configuration snapshot.

Progress bars and timestamped logs show progress through discovery, embedding, and distance calculation.

## Compute mAP (optional)
After generating results, compute Mean Average Precision from the flattened distances and series indices:
```bash
python metrics/map.py \
  --distances ./results/evaluation_results/pairwise_distances.npz \
  --series-indices ./results/series_to_indices.json \
  --output_csv ./results/metrics/map.csv
```

If you need to derive indices from labels and paths locally instead, provide `--labels` and `--image-paths` to `metrics/map.py` (using the saved `image_paths.json`), but be aware that sharing paths reveals filenames.
