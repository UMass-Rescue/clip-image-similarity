#!/usr/bin/env python3
"""Evaluate pretrained and finetuned checkpoints for one dataset.

Runs clip_image_similarity.cli + metrics.accuracy_hits_k for every
(model, checkpoint_type, epoch) combination, then writes a summary CSV
and learning-curve / final-epoch comparison plots.

Usage:
    python run_eval.py --config eval_config_cuhk.json [--overwrite]
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Eval run helpers
# ---------------------------------------------------------------------------

def _run_one(
    *,
    model_id: str,
    checkpoint_path: Optional[str],
    labels: str,
    input_dir: str,
    output_dir: str,
    eval_k: List[int],
    overwrite: bool,
    skip_existing: bool,
) -> None:
    out = Path(output_dir)
    metrics_csv = str(out / "metrics" / "accuracy_hits_k.csv")

    if skip_existing:
        if (out / "metrics" / "accuracy_hits_k.csv").is_file():
            tqdm.write(f"    skip (metrics csv exists)")
            return
        pairwise_done = (out / "evaluation_results" / "pairwise_distances.npz").is_file()
    else:
        pairwise_done = False

    if not pairwise_done:
        cli_cmd = [
            "python", "-m", "clip_image_similarity.cli",
            "--model", model_id,
            "--anonymize-labels", labels,
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--pairwise-dtype", "float16",
        ]
        if checkpoint_path:
            cli_cmd += ["--checkpoint_path", checkpoint_path]
        if overwrite:
            cli_cmd.append("--overwrite")
        subprocess.run(cli_cmd, check=True, stdin=subprocess.DEVNULL)
    else:
        tqdm.write(f"    skip cli (distances exist), recomputing metrics")

    acc_cmd = [
        "python", "-m", "metrics.accuracy_hits_k",
        "--pairwise-output-dir", output_dir,
        "--k", *[str(k) for k in eval_k],
        "--output-csv", metrics_csv,
    ]
    subprocess.run(acc_cmd, check=True, stdin=subprocess.DEVNULL)


def _build_run_list(cfg: Dict[str, Any]) -> List[Dict]:
    """Return flat list of run descriptors for the configured dataset."""
    dataset_name = cfg["dataset"]
    input_dir = cfg["series_image_path"]
    step1_base = cfg["step1_base"]
    step2_base = cfg["step2_base"]
    step3_base = cfg["step3_base"]
    series_lbl_sub = cfg["series_labels"]
    subseries_lbl_sub = cfg["subseries_labels"]

    runs = []
    for model in cfg["models"]:
        mn = model["name"]
        step1_dir = f"{step1_base}/{mn}/{dataset_name}"
        step2_model = f"{step2_base}/{mn}/{dataset_name}"
        step3_model = f"{step3_base}/{mn}/{dataset_name}"
        series_lbl = f"{step1_dir}/{series_lbl_sub}"
        subseries_lbl = f"{step1_dir}/{subseries_lbl_sub}"

        # pretrained baseline
        runs.append({
            "label": f"{mn} / pretrained",
            "model_id": model["id"],
            "checkpoint_path": None,
            "labels": series_lbl,
            "input_dir": input_dir,
            "output_dir": f"{step3_model}/pretrained",
            "model_name": mn,
            "checkpoint_type": "pretrained",
            "epoch": None,
        })

        # series epochs
        for epoch in cfg["eval_epochs"]:
            runs.append({
                "label": f"{mn} / series / epoch_{epoch}",
                "model_id": model["id"],
                "checkpoint_path": f"{step2_model}/{dataset_name}-series/checkpoints/epoch_{epoch}.pt",
                "labels": series_lbl,
                "input_dir": input_dir,
                "output_dir": f"{step3_model}/series/epoch_{epoch}",
                "model_name": mn,
                "checkpoint_type": "series",
                "epoch": epoch,
            })

        # subseries epochs
        for epoch in cfg["eval_epochs"]:
            runs.append({
                "label": f"{mn} / subseries / epoch_{epoch}",
                "model_id": model["id"],
                "checkpoint_path": f"{step2_model}/{dataset_name}-subseries/checkpoints/epoch_{epoch}.pt",
                "labels": subseries_lbl,
                "input_dir": input_dir,
                "output_dir": f"{step3_model}/subseries/epoch_{epoch}",
                "model_name": mn,
                "checkpoint_type": "subseries",
                "epoch": epoch,
            })

    return runs


# ---------------------------------------------------------------------------
# Summary / aggregation
# ---------------------------------------------------------------------------

def _read_accuracy_csv(path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["k"])] = float(row["accuracy_at_k"])
    return out


def _collect_results(runs: List[Dict], step3_base: str) -> List[Dict]:
    """Read accuracy CSVs from completed runs and return rows for the summary CSV."""
    rows = []
    for run in runs:
        csv_path = Path(run["output_dir"]) / "metrics" / "accuracy_hits_k.csv"
        if not csv_path.is_file():
            tqdm.write(f"  WARNING: missing {csv_path} — skipping.")
            continue
        accuracy_by_k = _read_accuracy_csv(csv_path)
        for k, acc in accuracy_by_k.items():
            rows.append({
                "model": run["model_name"],
                "checkpoint_type": run["checkpoint_type"],
                "epoch": run["epoch"] if run["epoch"] is not None else "pretrained",
                "k": k,
                "accuracy_at_k": acc,
            })
    return rows


def _write_summary_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "checkpoint_type", "epoch", "k", "accuracy_at_k"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary CSV: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(rows: List[Dict], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = sorted({r["model"] for r in rows})
    ks = sorted({r["k"] for r in rows})

    # Index rows for quick lookup: (model, checkpoint_type, epoch, k) -> accuracy
    idx: Dict = {}
    for r in rows:
        idx[(r["model"], r["checkpoint_type"], r["epoch"], r["k"])] = r["accuracy_at_k"]

    _plot_learning_curves(idx, models, ks, plots_dir)


def _plot_learning_curves(idx, models, ks, plots_dir: Path) -> None:
    """Single stacked figure: rows=models, cols=k values, tight per-subplot y-axis."""
    epochs_int = sorted({ep for (_, _, ep, _) in idx if isinstance(ep, int)})
    n_rows, n_cols = len(models), len(ks)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        sharey=False,
    )
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[row] for row in axes]

    for row_i, model in enumerate(models):
        model_short = model.rsplit("/", 1)[-1]
        for col_j, k in enumerate(ks):
            ax = axes[row_i][col_j]
            is_top_right = row_i == 0 and col_j == n_cols - 1

            series_vals = [idx.get((model, "series", ep, k)) for ep in epochs_int]
            subseries_vals = [idx.get((model, "subseries", ep, k)) for ep in epochs_int]
            pretrained_val = idx.get((model, "pretrained", "pretrained", k))

            if any(v is not None for v in series_vals):
                ax.plot(epochs_int, series_vals, marker="o", color="steelblue", label="series")
            if any(v is not None for v in subseries_vals):
                ax.plot(epochs_int, subseries_vals, marker="s", color="darkorange", label="subseries")
            if pretrained_val is not None:
                ax.axhline(pretrained_val, linestyle="--", color="gray", label="pretrained")

            all_vals = [v for v in [*series_vals, *subseries_vals, pretrained_val] if v is not None]
            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                margin = max(0.003, (hi - lo) * 0.2)
                ax.set_ylim(max(0.0, lo - margin), min(1.0, hi + margin))

            ax.set_title(f"k={k}", fontsize=10)
            ax.set_xlabel("epoch", fontsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(labelsize=8)

            if col_j == 0:
                ax.set_ylabel(f"{model_short}\nAccuracy@k", fontsize=8)
            if is_top_right:
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Learning curves", fontsize=13, fontweight="bold")
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = plots_dir / f"learning_curves_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints for one dataset and summarise results."
    )
    parser.add_argument("--config", default="eval_config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a run entirely if its metrics CSV exists; skip only cli if distances exist.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    runs = _build_run_list(cfg)

    print(f"Dataset: {cfg['dataset']}  |  {len(runs)} eval runs")

    for run in tqdm(runs, desc="Eval runs"):
        tqdm.write(f"  {run['label']}")
        try:
            _run_one(
                model_id=run["model_id"],
                checkpoint_path=run["checkpoint_path"],
                labels=run["labels"],
                input_dir=run["input_dir"],
                output_dir=run["output_dir"],
                eval_k=cfg["eval_k"],
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
            )
        except subprocess.CalledProcessError as e:
            tqdm.write(f"\nERROR: run failed for {run['label']} (exit {e.returncode})", file=sys.stderr)
            sys.exit(e.returncode)

    print("\nAll runs complete. Generating summary...")

    rows = _collect_results(runs, cfg["step3_base"])
    if not rows:
        print("No results to summarise.")
        return

    summary_path = Path(cfg["step3_base"]) / f"summary_{cfg['dataset']}.csv"
    _write_summary_csv(rows, summary_path)

    plots_dir = Path(cfg["step3_base"]) / "plots" / cfg["dataset"]
    _make_plots(rows, plots_dir)


if __name__ == "__main__":
    main()
