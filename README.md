# Clip Image Similarity

[![codecov](https://codecov.io/gh/UMass-Rescue/clip-image-similarity/graph/badge.svg?token=UI639IVPVS)](https://codecov.io/gh/UMass-Rescue/clip-image-similarity) [![Tests](https://github.com/UMass-Rescue/clip-image-similarity/actions/workflows/tests.yml/badge.svg)](https://github.com/UMass-Rescue/clip-image-similarity/actions/workflows/tests.yml)

*A CLIP-based toolkit for embedding image folders, computing pairwise distances, and evaluating image-series retrieval.*

## Features

* **CLIP-based image embedding** with any OpenCLIP model (default: Apple's `DFN5B-CLIP-ViT-H-14-384`), with support for fine-tuned checkpoints.
* **GPU-accelerated** batch inference.
* **Compact pairwise distance storage**: flattened upper-triangular matrix in `float32`/`float16`, or per-image top-k neighbors.
* **Privacy-preserving** label anonymization (series → image-index mapping that hides filenames).
* **Retrieval metrics**: mean Average Precision (mAP@k), Precision@k, Recall@k — dataset-wide and per-series.
* **Distance distribution analysis**: within-series vs out-of-series histograms, with optional rendered image-pair samples grouped by similarity percentile.
* **Series cleanup tools**: filter out outlier images using mean within-series and out-of-series distance thresholds.
* **Sub-series clustering**: split each existing series into thresholded clusters of similar images.

## Installation

`make install` creates a local `.venv/` and installs all dependencies:

```bash
make install
```

`make run`, `make anonymize-labels`, and `make test` reuse this environment automatically. You only need to activate the venv when running the downstream metric scripts (sections 2–6 below) directly:

```bash
source .venv/bin/activate
```

## Tests

```bash
make test
```

Runs the `pytest` suite with coverage over `clip_image_similarity` and `metrics`.

## Pipeline overview

```
labels.json + /path/to/images
        │
        ▼
   make run                   ──▶  pairwise_distances.npz
   (embeds + anonymizes             image_paths.json
    in one step)                    series_to_indices.json
                                            │
                                            ├──▶ metrics.map                         (mAP@k CSV)
                                            ├──▶ metrics.precision_recall_plot       (P@k / R@k curves)
                                            ├──▶ metrics.distance_histogram          (histograms + sample renders)
                                            ├──▶ scripts.filter_series_by_avg_distance  (cleanup)
                                            └──▶ scripts.cluster_series_by_distance     (sub-series)
```

## 1. Compute pairwise distances (`make run`)

`make run` embeds every image under `INPUT_DIR`, writes pairwise distances under `OUTPUT_DIR`, and — when `ANONYMIZE_LABELS` is set — also writes the index-mapped `series_to_indices.json` required by every downstream metric.

**Default path (recommended):**

```bash
make run \
  INPUT_DIR=/path/to/images \
  OUTPUT_DIR=./out/run1 \
  MODEL="ViT-H-14-378-quickgelu" \
  PRETRAINED=dfn5b \
  ANONYMIZE_LABELS=/path/to/labels.json \
  BATCH_SIZE=64 \
  DEVICE=cuda \
  PAIRWISE_DTYPE=float16
```

**Fine-tuned checkpoint** — same as above plus `CHECKPOINT_PATH` (use the same `MODEL` / `PRETRAINED` that the checkpoint was trained against):

```bash
make run \
  INPUT_DIR=/path/to/images \
  OUTPUT_DIR=./out/run1_finetuned \
  MODEL="ViT-H-14-378-quickgelu" \
  PRETRAINED=dfn5b \
  CHECKPOINT_PATH=/path/to/checkpoints/epoch_27.pt \
  ANONYMIZE_LABELS=/path/to/labels.json \
  BATCH_SIZE=64 \
  DEVICE=cuda \
  PAIRWISE_DTYPE=float16
```

`labels.json` is a JSON object `{ series_name: [image_path, ...] }`. Paths are matched against the images discovered under `INPUT_DIR`.

### `make run` variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `INPUT_DIR` | yes | — | Root directory containing images (searched recursively). |
| `OUTPUT_DIR` | yes | — | Directory where results will be written. |
| `MODEL` | yes | — | OpenCLIP model id. Use the training-time name when combining with `PRETRAINED`/`CHECKPOINT_PATH`. For a quick start without fine-tuning, `MODEL=hf-hub:apple/DFN5B-CLIP-ViT-H-14-384` is a good default and does not require `PRETRAINED`. |
| `PRETRAINED` | no | none | OpenCLIP pretrained-weights identifier, e.g. `dfn5b`. |
| `CHECKPOINT_PATH` | no | none | Local checkpoint loaded after model creation, for fine-tuned models. |
| `BATCH_SIZE` | no | `32` | Batch size for embedding computation. |
| `DEVICE` | no | auto (CUDA if available) | `cuda`, `cuda:0`, `cpu`, etc. |
| `PAIRWISE_DTYPE` | no | `float32` | Storage precision: `float32` or `float16`. |
| `TOP_K` | no | none | Save only the top-k neighbors per image instead of the full distance matrix. |
| `ANONYMIZE_LABELS` | no | none | Path to `labels.json`; produces `series_to_indices.json`. Strongly recommended — required by every downstream metric script. |
| `OVERWRITE` | no | unset | Set to any non-empty value to allow overwriting existing output files. |

### Outputs of `make run`

| File | Description |
| --- | --- |
| `evaluation_results/pairwise_distances.npz` | Flattened upper-triangular distances `(1 - cosine_similarity)`. dtype controlled by `PAIRWISE_DTYPE`. Written when `TOP_K` is *not* set. |
| `evaluation_results/pairwise_topk.npz` | Per-image top-k neighbor indices/distances. Written when `TOP_K` is set, in place of `pairwise_distances.npz`. |
| `image_paths.json` | Ordered list of image paths corresponding to indices in the flattened array. **Do not share if filenames are sensitive.** |
| `series_to_indices.json` | Written when `ANONYMIZE_LABELS` is set. Maps series → indices for downstream evaluation while keeping paths private. |
| `config.json` | Snapshot of the run configuration. |
| `run.log` | Log of the run. |

> If you forgot to pass `ANONYMIZE_LABELS` to `make run`, you can generate `series_to_indices.json` after the fact with `make anonymize-labels OUTPUT_DIR=./out/run1 LABELS=/path/to/labels.json` (add `OVERWRITE=1` to replace an existing file).

## 2. Compute mAP

`metrics.map` produces per-series and mean mAP@k from the pairwise distances and the anonymized labels:

```bash
python -m metrics.map \
  --distances ./out/run1/evaluation_results/pairwise_distances.npz \
  --series-indices ./out/run1/series_to_indices.json \
  --output_csv ./out/run1/metrics/map.csv
```

**With top-k neighbors** (only valid if you ran with `TOP_K`):

```bash
python -m metrics.map \
  --topk ./out/run1/evaluation_results/pairwise_topk.npz \
  --series-indices ./out/run1/series_to_indices.json \
  --output_csv ./out/run1/metrics/map.csv
```

> `TOP_K` must be at least `largest_series_size - 1`, otherwise positives are missed and the script raises an error. Either re-run `make run` with a larger `TOP_K`, or use the full `pairwise_distances.npz` (don't set `TOP_K`).

Add `--labeled-images-only` to restrict the candidate pool to the union of labeled images instead of every image in the matrix.

## 3. Plot Precision@k and Recall@k curves

`metrics.precision_recall_plot` produces precision/recall curves and a precision-vs-recall plot, both dataset-wide and one per series:

```bash
python -m metrics.precision_recall_plot \
  --pairwise-output-dir ./out/run1
```

### Options

| Flag | Description |
| --- | --- |
| `--max-predictions N` | Maximum k to evaluate (inclusive). Defaults to `N-1`. |
| `--steps S` | Step size between consecutive k values (default `1`). Mutually exclusive with `--num-k`. |
| `--num-k K` | Sample roughly K log-spaced k values between 1 and `--max-predictions`. Useful for very large matrices. |

### Outputs

```
<OUTPUT_DIR>/graphs/precision_recall_<timestamp>/
├── all_series.png                          # P and R vs k (dataset-level)
├── all_series_precision_vs_recall.png      # P vs R curve (dataset-level)
└── series/
    └── <series_name>.png                   # one P/R-vs-k plot per series
```

This script requires `evaluation_results/pairwise_distances.npz` and `series_to_indices.json`. Top-k input is **not** supported here.

## 4. Plot distance histograms (and optional sample renderings)

`metrics.distance_histogram` plots overlayed histograms of cosine distances **within series** vs **out-of-series**, and can optionally render representative image-pair samples grouped into similarity-percentile bins.

Minimal:

```bash
python -m metrics.distance_histogram \
  --pairwise-output-dir ./out/run1
```

Finer bins, density normalization, plus similarity samples:

```bash
python -m metrics.distance_histogram \
  --pairwise-output-dir ./out/run1 \
  --bins 500 \
  --range-max 1.0 \
  --density \
  --save-similarity-samples \
  --samples-per-bin 5 \
  --num-percentile-bins 20
```

### Options

| Flag | Description |
| --- | --- |
| `--bins N` | Number of equal-width histogram bins (default `100`). |
| `--range-min` / `--range-max` | Distance range to plot (defaults `0.0` / `2.0`). |
| `--density` | Plot probability density instead of counts. |
| `--log-y` | Use a log y-axis. |
| `--labeled-images-only` | Restrict out-of-series candidates to the union of labeled images instead of every other image. |
| `--save-raw-values` | Also save the raw within/out-of-series distance arrays per series under `raw_values/<series>.npz`. |
| `--save-similarity-samples` | Render image-pair PNGs grouped by similarity percentile, with manifests. |
| `--samples-per-bin N` | Max image pairs to render per percentile bin (default `3`, with `--save-similarity-samples`). |
| `--num-percentile-bins N` | Number of percentile bins per pool (default `10`, with `--save-similarity-samples`). |

Histogram PNGs are written under `<OUTPUT_DIR>/graphs/distance_histogram_<timestamp>/`. Sample renderings (when enabled) are written under `<OUTPUT_DIR>/similarity_samples/<timestamp>/{within_series,out_of_series}/pct_<lo>_<hi>/`.

## 5. Filter series by average distance

`scripts.filter_series_by_avg_distance` cleans up noisy series labels: for each labeled image it computes the mean within-series and mean out-of-series distance, and drops images whose mean within-series distance is too high or whose mean out-of-series distance is too low. Series that end up with fewer than 2 retained images are dropped.

```bash
python -m scripts.filter_series_by_avg_distance \
  --pairwise-output-dir ./out/run1 \
  --in-series-threshold 0.35 \
  --out-of-series-threshold 0.55
```

The script writes a new sub-directory under `OUTPUT_DIR`, named after the thresholds (e.g. `filtered_series_by_avg_distance_in_t0_35_out_t0_55/`), containing:

* `series_to_indices.json` — filtered series → indices. Drop-in replacement for downstream metrics: point any of the metric scripts above at this directory by passing it as `--pairwise-output-dir`, or by passing the file directly to `--series-indices` for `metrics.map`.
* `series_to_image_paths.json` — same data but as series → image paths.

A summary of what was kept vs. removed is logged to stdout.

## 6. Cluster series into sub-series

`scripts.cluster_series_by_distance` splits each existing series into sub-series using agglomerative clustering over the saved cosine distances. Clustering is performed independently within each original series; images from different original series are never merged.

```bash
python -m scripts.cluster_series_by_distance \
  --pairwise-output-dir ./out/run1 \
  --distance-threshold 0.25 \
  --min-samples 2
```

By default this uses average linkage and automatically cuts the hierarchy at `--distance-threshold`. You can pass `--linkage complete` for stricter clusters where the threshold acts on the maximum pairwise distance between clusters, or `--linkage single` for nearest-link behavior.

The script requires the full `evaluation_results/pairwise_distances.npz`; top-k output is not enough for clustering because arbitrary within-series pair distances are needed.

The script writes a new sub-directory under `OUTPUT_DIR`, named after the threshold, linkage, and minimum size (for example `clustered_series_by_distance_t0_25_average_min2/`), containing:

* `series_to_indices.json` — the primary output. This is a flat mapping of output series labels to image indices, and can be passed to downstream tools that accept a series mapping. If an original series produces only one retained sub-series, its original series name is preserved. If it produces multiple retained sub-series, labels use suffixes such as `original_series_subseries_0`.
* `series_to_image_paths.json` — the same flat sub-series mapping as image paths.
* `series_to_subseries_indices.json` — nested mapping from original series to sub-series labels to indices.
* `summary.json` — aggregate, per-series, and per-sub-series image counts, including how many small sub-series were dropped by `--min-samples`.

## Performance considerations

### Batch size

Start small (`BATCH_SIZE=16` or `32`) and scale up while watching GPU memory. For reference, `BATCH_SIZE=256` reaches ~81% VRAM utilization on an RTX 5090 (32 GB) embedding 30K images.

### Precision

`PAIRWISE_DTYPE=float16` halves the size of `pairwise_distances.npz` with negligible impact on retrieval accuracy. Use the default `float32` if you need higher precision.

### Top-k mode

For very large datasets, `TOP_K=K` saves only the `K` nearest neighbors per image instead of the full distance matrix, dramatically reducing storage when `K << N`. Trade-offs:

* `metrics.precision_recall_plot` requires the full distance matrix and **cannot** consume top-k output.
* `metrics.map` works with top-k, but `K` must be at least `largest_series_size - 1`.

If unsure, leave `TOP_K` unset and use `PAIRWISE_DTYPE=float16` instead — that already produces a compact file while keeping every downstream tool usable.

## Benchmarks

See [BENCHMARK.md](BENCHMARK.md) for detailed timing breakdowns, resource usage, and throughput on the PIPA dataset.
