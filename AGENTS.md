# AGENTS GUIDE

## Purpose
- Embed a folder of images with a CLIP model, compute pairwise distances, and export results for series retrieval evaluation (flattened upper-tri array or per-image top-k neighbors).
- Optional: map provided series labels (paths) to indices for privacy-preserving mAP computation.
- mAP is computed offline via `metrics/map.py` using either the flattened distances or top-k neighbors plus series indices.

## Coding Practices
- Keep functions small, well-documented, and modular; each function has a concise purpose line plus clear args/returns in docstrings.
- Prefer explicit, deterministic ordering (sorted image list) to align indices with outputs.
- Avoid hidden state: pass data explicitly between steps (e.g., top-k extraction returns arrays, then saved).
- Use `allow_pickle=False` when saving/loading npz for safety.
- Default to float32; allow float16 for storage to save space. Use canonical POSIX paths when writing path lists.
- Logs via a simple `log()` with timestamps; avoid logging sensitive paths to disk unless required.
- Compilation checks via `python -m compileall` are used as a lightweight lint.

## CLI (`clip_image_similarity/cli.py`)
- Parses required run params (input/output/model/device/batch size) plus storage dtype (`--pairwise-dtype`) and optional top-k output (`--top-k`).
- Discovers images (sorted), embeds them, computes cosine similarity -> distances.
- Outputs:
  - Flattened upper-tri distances (`evaluation_results/pairwise_distances.npz`) or top-k neighbors (`pairwise_topk.npz`).
  - `image_paths.json`: ordered list of paths (sensitive; don’t share).
  - Optional `series_to_indices.json` if `--anonymize-labels` is provided.
  - `config.json`.
- Overwrite protection unless `--overwrite` is set.

## Distances & Access Helpers
- `packed_distances.py`: flatten full distance matrix, infer N, and access `distance(i, j)` via a packed upper-tri array.
- `topk.py`: extract per-row top-k neighbors (indices + distances 2D arrays), save to npz with dtype metadata, and load/access via `TopKNeighbors`.

## Labels
- `labels.py`: map series labels (paths) to indices matching the embedding order; errors if any label path is missing.

## mAP (`metrics/map.py`)
- Supports two inputs: flattened distances (`--distances`) or top-k neighbors (`--topk`).
- Uses series->indices JSON (or derives indices from labels + image_paths.json).
- Builds rankings (from packed distances or stored top-k) and computes mAP@k per series; outputs CSV with per-series and mean mAP.
- Optional `--labeled-images-only` restricts candidates to labeled images.

## Outputs & Privacy
- Flattened/top-k distance artifacts are path-free; `image_paths.json` reveals paths and should not be shared if sensitive.
- `series_to_indices.json` contains only indices and is safe to share.
