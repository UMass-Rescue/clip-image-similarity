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

Outputs:
- `file_to_id_map/anon_id_map.json`: absolute path to anonymous numeric ID map (**DO NOT SHARE THIS FILE IF YOU NEED TO KEEP FILE NAMES PRIVATE**).
- `evaluation_results/pairwise_clip_compare.json`: all unordered pairwise distances `(1 - cosine_similarity)` using only anonymous IDs.
- `config.json`: run configuration snapshot

Progress bars and timestamped logs show progress through discovery, embedding, and distance calculation.

## Output Details and Privacy
- `file_to_id_map/anon_id_map.json`: Maps absolute file paths to anonymous IDs. **Do not share this file** if you need to keep file names private.
- `evaluation_results/pairwise_clip_compare.json`: Pairwise distances using only anonymous IDs. **This file alone is sufficient to share results without revealing paths.**
- `config.json`: Snapshot of the run configuration (paths, model id, etc.).
