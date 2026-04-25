import json
from pathlib import Path

import numpy as np
import pytest

from metrics.precision_recall_plot import PrecisionRecallPlotRunner


def _flatten_upper_triangle_numpy(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(m[i, j]))
    return np.asarray(out, dtype=np.float32)


def test_precision_recall_plot_runner_writes_pngs(tmp_path: Path):
    pytest.importorskip("matplotlib")

    # Build a 4x4 distance matrix where within-series mates are closest.
    # Series A: {0,1}, Series B: {2,3}
    m = np.array(
        [
            [0.0, 0.1, 0.9, 1.0],
            [0.1, 0.0, 0.9, 1.0],
            [0.9, 0.9, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    flat = _flatten_upper_triangle_numpy(m)

    out_dir = tmp_path / "run_output"
    eval_dir = out_dir / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        eval_dir / "pairwise_distances.npz",
        distances=flat,
        dtype="float32",
        allow_pickle=False,
    )

    series_to_indices = {"A": [0, 1], "B": [2, 3]}
    (out_dir / "series_to_indices.json").write_text(
        json.dumps(series_to_indices), encoding="utf-8"
    )

    runner = PrecisionRecallPlotRunner(max_predictions=2)
    paths = runner.run(pairwise_output_dir=out_dir)

    assert paths.dataset_plot.is_file()
    assert paths.dataset_precision_vs_recall_plot.is_file()
    assert (paths.per_series_dir / "A.png").is_file()
    assert (paths.per_series_dir / "B.png").is_file()


