# Fine-Tune and Test Workflow

This workflow uses one config file to prepare data, fine-tune a fixed model, and evaluate the resulting checkpoints on the held-out series-level test set.

## Variables Used Below

Replace every placeholder in angle brackets with the matching value from `finetune_config.json`.

The four fields in `finetune_config.json` are:

- `dataset_name`: an identifier for this dataset and fine-tuning run.
- `image_dir`: the directory containing all images in the dataset.
- `labels_json`: a JSON mapping that lists the series and the images belonging to each series.
- `output_dir`: the directory where this workflow writes all generated configs, processed data, checkpoints, evaluation files, and plots.

```text
<dataset_name> = value of dataset_name in finetune_config.json
<image_dir>    = value of image_dir in finetune_config.json
<labels_json>  = value of labels_json in finetune_config.json
<output_dir>   = value of output_dir in finetune_config.json
```

For example, if `finetune_config.json` contains:

```json
{
  "output_dir": "/data/run1"
}
```

then this command:

```bash
bash <output_dir>/configs/run_finetune.sh
```

becomes:

```bash
bash /data/run1/configs/run_finetune.sh
```

After `scripts/prepare_finetune_workflow.py` runs, `<output_dir>/configs/commands.txt` contains the same workflow commands with concrete generated paths.

## Input Config

Create `finetune_config.json` from the example:

```bash
cp finetune_config.example.json finetune_config.json
```

Edit the four fields:

```json
{
  "dataset_name": "<dataset_name>",
  "image_dir": "<image_dir>",
  "labels_json": "<labels_json>",
  "output_dir": "<output_dir>"
}
```

`labels_json` must be a JSON object mapping each series name to the image paths in that series:

```json
{
  "series_001": [
    "/absolute/path/to/image_001.jpg",
    "/absolute/path/to/image_002.jpg"
  ],
  "series_002": [
    "/absolute/path/to/image_003.jpg",
    "/absolute/path/to/image_004.jpg"
  ]
}
```

The image paths in `labels_json` must point to images under `image_dir`.

If your dataset is organized as one folder per series, generate `labels_json` with:

```bash
python scripts/generate_labels_json.py \
  --series-dir <image_dir> \
  --output-json <labels_json>
```

For example, if `image_dir` is `/home/gbiss/Documents/image-series-dataset/series`,
each immediate child directory becomes one series in the generated JSON.

## 1. Prepare Data

Clone and set up the evaluation repository:

```bash
git clone https://github.com/UMass-Rescue/clip-image-similarity.git
cd clip-image-similarity
git checkout finetune_test
make install
source .venv/bin/activate
```

Create and edit the workflow config:

```bash
cp finetune_config.example.json finetune_config.json
# edit finetune_config.json
```

Generate the internal workflow configs:

```bash
python scripts/prepare_finetune_workflow.py --config finetune_config.json
```

The preparation script first checks that every labeled image can be decoded by
Pillow. The generated data config points at:

```text
<output_dir>/configs/decode_checked_series_labels.json
```

Any images that fail decoding are omitted from that file and recorded in:

```text
<output_dir>/configs/image_decode_failures.json
```

Images that decode successfully but trigger Pillow warnings are retained and
listed in the same manifest under `decode_warnings`.

Optionally, have the preparation script create a de-duplicated label mapping
after the decode precheck:

```bash
python scripts/prepare_finetune_workflow.py --config finetune_config.json --deduplicate-images
```

This uses a 64-bit perceptual hash (`pHash`) and treats images whose hashes are
within a Hamming distance of 4 as duplicates by default. To tune that threshold:

```bash
python scripts/prepare_finetune_workflow.py --config finetune_config.json --deduplicate-images --phash-threshold 2
```

When de-duplication is enabled, the generated data config points at:

```text
<output_dir>/configs/deduplicated_series_labels.json
```

The removed-duplicate audit manifest is written to:

```text
<output_dir>/configs/deduplicated_images.json
```

Run data preparation:

```bash
python run_data_workflows.py --config <output_dir>/configs/data_config.json
```

This writes processed data under:

```text
<output_dir>/step1_processed/
```

During data preparation, the workflow still writes a loadable copy of the label mapping and a skipped-image manifest:

```text
<output_dir>/step1_processed/<model_name>/<dataset_name>/data/loadable_series.json
<output_dir>/step1_processed/<model_name>/<dataset_name>/data/skipped_images.json
```

The train/val/test splits are created from `loadable_series.json`, so later workflow stages only see images that Pillow can decode.

## 2. Fine-Tune

Clone and set up the fine-tuning repository:

```bash
git clone https://github.com/UMass-Rescue/2026-project-open-clip-sem-hash.git
cd 2026-project-open-clip-sem-hash
git checkout finetune_test

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-training.txt
```

Run fine-tuning from `src`:

```bash
cd src
bash <output_dir>/configs/run_finetune.sh
```

The workflow fine-tunes `hf-hub:timm/ViT-SO400M-16-SigLIP2-384` for 39 epochs and writes checkpoints under:

```text
<output_dir>/step2_finetune_logs/
```

For the default 39-epoch run, the generated fine-tuning command keeps the
learning-rate schedule for epochs 1 through 9 identical to the earlier 9-epoch
workflow, then uses a lower extension learning rate for epochs 10 through 39.

## 3. Evaluate Checkpoints

Return to the evaluation repository:

```bash
cd /path/to/clip-image-similarity
source .venv/bin/activate
```

Run test-set evaluation:

```bash
python run_eval.py --config <output_dir>/configs/test_eval_config.json
```

If `<output_dir>/step3_eval/` already contains pairwise distance outputs, rerunning the same command reuses those distances and regenerates the metrics CSVs and plots. Use `--overwrite` only when you want to recompute the embeddings and pairwise distances themselves. Use `--skip-existing` to leave runs alone when both metrics CSVs already exist.

The final outputs are:

```text
<output_dir>/step3_eval/summary_<dataset_name>.csv
<output_dir>/step3_eval/summary_precision_recall_<dataset_name>.csv
<output_dir>/step3_eval/summary_precision_at_k_<dataset_name>.csv
<output_dir>/step3_eval/plots/<dataset_name>/
```

The summary CSVs contain:

- `summary_<dataset_name>.csv`: Accuracy@k / Hits@k.
- `summary_precision_recall_<dataset_name>.csv`: Precision@k and Recall@k, including hit and denominator counts.
- `summary_precision_at_k_<dataset_name>.csv`: Precision@k only, as a compact CSV for precision-at-k analysis.

The plots folder contains learning-curve plots for Accuracy@k and Precision@k at the configured `eval_k` values, plus separate Precision-vs-Recall plots for series epochs and sub-series epochs. The pretrained baseline appears in both Precision-vs-Recall plots. Each point on those curves is a k value, and duplicate rows for the same model/checkpoint/epoch/k are averaged. All are evaluated against the same held-out test split.

Please send us the CSVs under `<output_dir>/step3_eval/` and the plots in `<output_dir>/step3_eval/plots/<dataset_name>/`.

## Troubleshooting

- If data preparation fails because an output directory already exists, use a fresh `output_dir`.
- If label filtering drops more images than expected, inspect `data/skipped_images.json` under the generated step1 dataset directory.
- If evaluation reports missing checkpoints, confirm that fine-tuning completed and that checkpoints exist under `<output_dir>/step2_finetune_logs/`.
- If a command contains `<output_dir>`, replace it with the actual `output_dir` value from `finetune_config.json`, or copy the concrete command from `<output_dir>/configs/commands.txt`.
