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

  ```bash
  make run \
    INPUT_DIR=/path/to/images \
    OUTPUT_DIR=/path/to/output \
    MODEL=hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
    BATCH_SIZE=16 \
    DEVICE=cuda \
    ANONYMIZE_LABELS=/path/to/labels.json \
    PAIRWISE_DTYPE=float16
  ```
*OR*
  ```bash
  poetry install
  poetry run clip-pairwise-eval \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
    --batch-size 16 \
    --device cuda \
    --anonymize-labels /path/to/labels.json \
    --pairwise-dtype float16
  ```

## Installation

**Using Makefile** (auto-creates venv and installs dependencies)
```bash
make install
source .venv/bin/activate
```

**Using Poetry**
```bash
poetry install
poetry shell
```

## Parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-dir`, `-i` | ✅ | - | Root directory containing images to process. |
| `--output-dir`, `-o` | ✅ | - | Directory where results will be written. |
| `--model`, `-m` | ❌ | `hf-hub:apple/DFN5B-CLIP-ViT-H-14-384` | Hugging Face Hub model ID for OpenCLIP. |
| `--batch-size`, `-b` | ❌ | `32` | Batch size for embedding computation. |
| `--device`, `-d` | ❌ | Auto (CUDA if available) | Device to run on (e.g., `cuda`, `cuda:0`, `cpu`). |
| `--pairwise-dtype` | ❌ | `float32` | Numeric precision for storing distances (`float32` or `float16`). |
| `--top-k` | ❌ | None | Save top-k neighbors per image instead of full flattened distances. |
| `--anonymize-labels` | ❌ | None | Path to labels JSON (series → image paths); converts to series → indices. |
| `--image-exts` | ❌ | Common formats | Comma-separated list of image extensions (e.g., `jpg,png,jpeg`). |
| `--overwrite` | ❌ | `false` | Allow overwriting existing output files. |

**Running the CLI:**
```bash
python -m clip_image_similarity.cli \
  --input-dir /path/to/images \
  --output-dir /path/to/output \
  --model hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  --batch-size 16 \
  --device cuda
```

Or use the Makefile wrapper:
```bash
make run INPUT_DIR=/path/to/images OUTPUT_DIR=/path/to/output
```

## Outputs

| File | Description |
| --- | --- |
| `evaluation_results/pairwise_distances.npz` | Flattened upper-triangular distances `(1 - cosine_similarity)`; dtype `float32` (default) or `float16` via `--pairwise-dtype`, saved with dtype metadata. |
| `evaluation_results/pairwise_topk.npz` | Emitted when `--top-k` is set; contains per-image neighbor indices/distances plus stored `top_k`, dtype, and index dtype metadata. |
| `image_paths.json` | Ordered list of image paths corresponding to indices in the flattened array. **DO NOT SHARE IF FILENAMES ARE SENSITIVE.** |
| `series_to_indices.json` | Optional; only written when `--anonymize-labels` is provided. Maps series -> list of indices for downstream mAP while keeping paths private. |
| `config.json` | Snapshot of the run configuration. |

## Generate Anonymous Labels (optional)

If you ran the CLI without `--anonymize-labels` but later want to generate `series_to_indices.json`, you can use the standalone script:

```bash
make anonymize-labels OUTPUT_DIR=./results LABELS=./path/to/labels.json
```

Or run directly:
```bash
python -m clip_image_similarity.generate_anonymous_labels \
  --output-dir ./results \
  --labels ./path/to/labels.json \
  --overwrite  # optional: overwrite existing series_to_indices.json
```

This reads `image_paths.json` from the output directory and generates `series_to_indices.json` using your provided labels file.

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
