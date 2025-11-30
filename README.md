# Clip Image Similarity

[![codecov](https://codecov.io/gh/UMass-Rescue/clip-image-similarity/graph/badge.svg?token=UI639IVPVS)](https://codecov.io/gh/UMass-Rescue/clip-image-similarity) [![Tests](https://github.com/UMass-Rescue/clip-image-similarity/actions/workflows/tests.yml/badge.svg)](https://github.com/UMass-Rescue/clip-image-similarity/actions/workflows/tests.yml)

*A CLIP-based toolkit for embedding image folders and generating compact pairwise distance matrices for retrieval and evaluation.*

## Features

* 🔍 **CLIP-based image embedding** (any OpenCLIP model, default is Apple's DFN5B-CLIP-ViT-H-14-384)
* ⚡ **GPU-accelerated** batch inference
* 📦 **Compact flattened pairwise distance arrays** (upper-triangular matrix, float32/float16 storage, top-k neighbors)
* 🔒 **Privacy-preserving** series label anonymization helper
* 📊 **Mean Average Precision (mAP) computation** from either flattened distances or stored top-k neighbors

## Quickstart

Install dependencies (venv, Poetry, or system) and run the CLI:
  ```bash
  python -m clip_image_similarity.cli \
    --input-dir /path/to/images \
    --output-dir /tmp/clip-results \
    --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
    --batch-size 16 \
    --device cuda
  ```

## How to Run

**Option 1: Makefile (auto-creates venv and installs deps)**
```bash
make run \
  INPUT_DIR=/path/to/images \
  OUTPUT_DIR=/tmp/clip-results \
  MODEL=hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  BATCH_SIZE=16 \
  DEVICE=cuda \
  TOP_K=1000 \
  OVERWRITE=1
```

**Option 2: Create venv with Makefile, then run Python manually**
```bash
make install                     # sets up .venv and installs requirements
source .venv/bin/activate
python -m clip_image_similarity.cli \
  --input-dir /path/to/images \
  --output-dir /tmp/clip-results \
  --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  --batch-size 16 \
  --device cuda \
  --top-k 1000  # optional: save per-image top-k neighbors instead of full flattened distances
# OR
python -m clip_image_similarity.cli --input-dir ./resources/images --output-dir ./results_3 --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 --batch-size 16 
```

**Option 3: Poetry**
```bash
poetry install
poetry run clip-pairwise-eval \
  --input-dir /path/to/images \
  --output-dir /tmp/clip-results \
  --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  --device cpu \
  --top-k 1000
# OR
poetry run python -m clip_image_similarity.cli --input-dir ./resources/images --output-dir ./results --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384
```

## Outputs

| File | Description |
| --- | --- |
| `evaluation_results/pairwise_distances.npz` | Flattened upper-triangular distances `(1 - cosine_similarity)`; dtype `float32` (default) or `float16` via `--pairwise-dtype`, saved with `allow_pickle=False` and dtype metadata. |
| `evaluation_results/pairwise_topk.npz` | Emitted when `--top-k` is set; contains per-image neighbor indices/distances plus stored `top_k`, dtype, and index dtype metadata. |
| `image_paths.json` | Ordered list of image paths corresponding to indices in the flattened array. **Do not share if filenames are sensitive.** |
| `series_to_indices.json` | Optional; only written when `--anonymize-labels` is provided. Maps series -> list of indices for downstream mAP while keeping paths private. |
| `config.json` | Snapshot of the run configuration for auditing/repro. |

Progress bars and timestamped logs show progress through discovery, embedding, and distance calculation.

## Compute mAP (optional)

After generating results, compute Mean Average Precision from the flattened distances and series indices:
```bash
python -m metrics.map \
  --distances ./results/evaluation_results/pairwise_distances.npz \
  --series-indices ./results/series_to_indices.json \
  --output_csv ./results/metrics/map.csv
```

If you saved top-k neighbors instead of the full flattened distances:
```bash
python -m metrics.map \
  --topk ./results/evaluation_results/pairwise_topk.npz \
  --series-indices ./results/series_to_indices.json \
  --output_csv ./results/metrics/map.csv
```

If you need to derive indices from labels and paths locally instead, provide `--labels` and `--image-paths` to `metrics/map.py` (using the saved `image_paths.json`), but be aware that sharing paths reveals filenames:
```bash
python -m metrics.map \
  --distances ./results/evaluation_results/pairwise_distances.npz \
  --labels ./resources/labels/images_series_labels.json \
  --image-paths ./results/image_paths.json \
  --output_csv ./results/metrics/map.csv
```
