#!/usr/bin/env python3
"""Evaluate pretrained and finetuned checkpoints for one dataset.

Runs clip_image_similarity.cli plus retrieval metrics for every
(model, checkpoint_type, epoch) combination, then writes summary CSVs
and learning-curve plots.

Usage:
    python run_eval.py --config eval_config_cuhk.json [--overwrite] [--skip-existing]
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
    metrics_dir = out / "metrics"
    pairwise_path = out / "evaluation_results" / "pairwise_distances.npz"
    accuracy_csv = str(metrics_dir / "accuracy_hits_k.csv")
    precision_recall_csv = str(metrics_dir / "precision_recall_at_k.csv")

    if skip_existing:
        metrics_done = (
            (metrics_dir / "accuracy_hits_k.csv").is_file()
            and (metrics_dir / "precision_recall_at_k.csv").is_file()
        )
        if metrics_done:
            tqdm.write(f"    skip (metrics csvs exist)")
            return

    pairwise_done = pairwise_path.is_file() and not overwrite

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
        "--output-csv", accuracy_csv,
    ]
    subprocess.run(acc_cmd, check=True, stdin=subprocess.DEVNULL)

    pr_cmd = [
        "python", "-m", "metrics.precision_recall_at_k",
        "--pairwise-output-dir", output_dir,
        "--all-k",
        "--output-csv", precision_recall_csv,
    ]
    subprocess.run(pr_cmd, check=True, stdin=subprocess.DEVNULL)


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


def _read_precision_recall_csv(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["k"])] = {
                "hits": int(row["hits"]),
                "precision_denom": int(row["precision_denom"]),
                "recall_denom": int(row["recall_denom"]),
                "precision_at_k": float(row["precision_at_k"]),
                "recall_at_k": float(row["recall_at_k"]),
            }
    return out


def _collect_accuracy_results(runs: List[Dict]) -> List[Dict]:
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


def _collect_precision_recall_results(runs: List[Dict]) -> List[Dict]:
    """Read precision/recall CSVs from completed runs and return summary rows."""
    rows = []
    for run in runs:
        csv_path = Path(run["output_dir"]) / "metrics" / "precision_recall_at_k.csv"
        if not csv_path.is_file():
            tqdm.write(f"  WARNING: missing {csv_path} — skipping.")
            continue
        pr_by_k = _read_precision_recall_csv(csv_path)
        for k, values in pr_by_k.items():
            rows.append({
                "model": run["model_name"],
                "checkpoint_type": run["checkpoint_type"],
                "epoch": run["epoch"] if run["epoch"] is not None else "pretrained",
                "k": k,
                **values,
            })
    return rows


def _write_summary_csv(rows: List[Dict], out_path: Path, fieldnames: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary CSV: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(
    accuracy_rows: List[Dict],
    pr_rows: List[Dict],
    plots_dir: Path,
    eval_k: List[int],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    if accuracy_rows:
        _make_metric_learning_plot(
            rows=accuracy_rows,
            plots_dir=plots_dir,
            value_key="accuracy_at_k",
            ylabel="Accuracy@k",
            title="Accuracy@k learning curves",
            filename_prefix="learning_curves",
        )

    if pr_rows:
        precision_plot_rows = [r for r in pr_rows if r["k"] in set(eval_k)]
        if precision_plot_rows:
            _make_metric_learning_plot(
                rows=precision_plot_rows,
                plots_dir=plots_dir,
                value_key="precision_at_k",
                ylabel="Precision@k",
                title="Precision@k learning curves",
                filename_prefix="precision_at_k_learning_curves",
            )
        _plot_precision_vs_recall_curves(pr_rows, plots_dir)


def _make_metric_learning_plot(
    *,
    rows: List[Dict],
    plots_dir: Path,
    value_key: str,
    ylabel: str,
    title: str,
    filename_prefix: str,
) -> None:
    models = sorted({r["model"] for r in rows})
    ks = sorted({r["k"] for r in rows})

    idx: Dict = {}
    for r in rows:
        idx[(r["model"], r["checkpoint_type"], r["epoch"], r["k"])] = r[value_key]

    _plot_learning_curves(idx, models, ks, plots_dir, ylabel, title, filename_prefix)


def _plot_learning_curves(
    idx,
    models,
    ks,
    plots_dir: Path,
    ylabel: str,
    title: str,
    filename_prefix: str,
) -> None:
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
                ax.set_ylabel(f"{model_short}\n{ylabel}", fontsize=8)
            if is_top_right:
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = plots_dir / f"{filename_prefix}_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_precision_vs_recall_curves(rows: List[Dict], plots_dir: Path) -> None:
    """Plot precision vs recall, with one curve per checkpoint epoch."""
    grouped: Dict = {}
    for r in rows:
        key = (r["model"], r["checkpoint_type"], r["epoch"], r["k"])
        grouped.setdefault(key, []).append(r)

    series_by_run: Dict = {}
    for (model, checkpoint_type, epoch, k), group_rows in grouped.items():
        precision = sum(r["precision_at_k"] for r in group_rows) / len(group_rows)
        recall = sum(r["recall_at_k"] for r in group_rows) / len(group_rows)
        series_by_run.setdefault((model, checkpoint_type, epoch), []).append(
            (k, recall, precision)
        )

    for checkpoint_type in ("series", "subseries"):
        _plot_precision_vs_recall_for_checkpoint_type(
            series_by_run=series_by_run,
            target_checkpoint_type=checkpoint_type,
            plots_dir=plots_dir,
        )


def _plot_precision_vs_recall_for_checkpoint_type(
    *,
    series_by_run: Dict,
    target_checkpoint_type: str,
    plots_dir: Path,
) -> None:
    curves = {
        key: points
        for key, points in series_by_run.items()
        if key[1] in {"pretrained", target_checkpoint_type}
    }
    if not any(key[1] == target_checkpoint_type for key in curves):
        return

    models = sorted({model for model, _, _ in curves})
    fig, ax = plt.subplots(figsize=(9, 6.5))
    markers = {
        "pretrained": "x",
        "series": "o",
        "subseries": "s",
    }
    colors_by_type = {
        "series": plt.cm.Blues,
        "subseries": plt.cm.Oranges,
    }
    epochs_by_type = {
        target_checkpoint_type: sorted(
            {
                epoch
                for _, ckpt_type, epoch in curves
                if ckpt_type == target_checkpoint_type and isinstance(epoch, int)
            }
        )
    }

    for (model, curve_checkpoint_type, epoch), points in sorted(
        curves.items(), key=lambda item: _pr_curve_sort_key(item[0])
    ):
        points = sorted(points, key=lambda p: p[0])
        _, recall_vals, precision_vals = zip(*points)
        model_short = model.rsplit("/", 1)[-1]
        label_parts = []
        if len(models) > 1:
            label_parts.append(model_short)
        label_parts.append(curve_checkpoint_type)
        if isinstance(epoch, int):
            label_parts.append(f"epoch {epoch}")
        label = " ".join(label_parts)
        color = _pr_curve_color(
            checkpoint_type=curve_checkpoint_type,
            epoch=epoch,
            epochs_by_type=epochs_by_type,
            colors_by_type=colors_by_type,
        )
        ax.plot(
            recall_vals,
            precision_vals,
            marker=markers.get(curve_checkpoint_type, "o"),
            markersize=3,
            linewidth=1.4 if isinstance(epoch, int) else 1.8,
            linestyle="--" if curve_checkpoint_type == "pretrained" else "-",
            color=color,
            alpha=0.85,
            label=label,
        )

    title_type = "Series" if target_checkpoint_type == "series" else "Sub-Series"
    fig.suptitle(f"Precision vs Recall ({title_type} Epochs)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = plots_dir / f"precision_vs_recall_{target_checkpoint_type}_curves_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _pr_curve_sort_key(key):
    model, checkpoint_type, epoch = key
    type_order = {"pretrained": 0, "series": 1, "subseries": 2}
    epoch_order = -1 if epoch == "pretrained" else int(epoch)
    return (model, type_order.get(checkpoint_type, 99), epoch_order)


def _pr_curve_color(
    *,
    checkpoint_type: str,
    epoch,
    epochs_by_type: Dict,
    colors_by_type: Dict,
):
    if checkpoint_type == "pretrained":
        return "gray"
    epochs = epochs_by_type.get(checkpoint_type, [])
    if not isinstance(epoch, int) or not epochs:
        return None
    if len(epochs) == 1:
        position = 0.65
    else:
        position = 0.35 + 0.55 * (epochs.index(epoch) / (len(epochs) - 1))
    return colors_by_type[checkpoint_type](position)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints for one dataset and summarise results."
    )
    parser.add_argument("--config", default="eval_config.json")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings/distances even when existing pairwise outputs are present.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a run entirely if both metrics CSVs exist.",
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
            tqdm.write(
                f"\nERROR: run failed for {run['label']} (exit {e.returncode})",
                file=sys.stderr,
            )
            sys.exit(e.returncode)

    print("\nAll runs complete. Generating summary...")

    accuracy_rows = _collect_accuracy_results(runs)
    pr_rows = _collect_precision_recall_results(runs)
    if not accuracy_rows and not pr_rows:
        print("No results to summarise.")
        return

    step3_base = Path(cfg["step3_base"])
    if accuracy_rows:
        summary_path = step3_base / f"summary_{cfg['dataset']}.csv"
        _write_summary_csv(
            accuracy_rows,
            summary_path,
            ["model", "checkpoint_type", "epoch", "k", "accuracy_at_k"],
        )

    if pr_rows:
        pr_summary_path = step3_base / f"summary_precision_recall_{cfg['dataset']}.csv"
        _write_summary_csv(
            pr_rows,
            pr_summary_path,
            [
                "model",
                "checkpoint_type",
                "epoch",
                "k",
                "hits",
                "precision_denom",
                "recall_denom",
                "precision_at_k",
                "recall_at_k",
            ],
        )
        precision_rows = [
            {
                "model": r["model"],
                "checkpoint_type": r["checkpoint_type"],
                "epoch": r["epoch"],
                "k": r["k"],
                "hits": r["hits"],
                "precision_denom": r["precision_denom"],
                "precision_at_k": r["precision_at_k"],
            }
            for r in pr_rows
        ]
        precision_summary_path = step3_base / f"summary_precision_at_k_{cfg['dataset']}.csv"
        _write_summary_csv(
            precision_rows,
            precision_summary_path,
            [
                "model",
                "checkpoint_type",
                "epoch",
                "k",
                "hits",
                "precision_denom",
                "precision_at_k",
            ],
        )

    plots_dir = step3_base / "plots" / cfg["dataset"]
    _make_plots(accuracy_rows, pr_rows, plots_dir, cfg["eval_k"])


if __name__ == "__main__":
    main()
