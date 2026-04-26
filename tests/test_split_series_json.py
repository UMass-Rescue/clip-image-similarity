import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_series_json.py"
SPEC = importlib.util.spec_from_file_location("split_series_json", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

load_series_json = MODULE.load_series_json
split_series_mapping = MODULE.split_series_mapping
write_series_splits = MODULE.write_series_splits


def test_split_series_mapping_splits_whole_series_without_test():
    series_mapping = {
        "a": [1, 2],
        "b": [3],
        "c": [4],
        "d": [5, 6],
        "e": [7],
    }

    splits = split_series_mapping(
        series_mapping,
        train_pct=60,
        val_pct=40,
        seed=7,
    )

    assert set(splits) == {"train", "val"}
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 2
    assert _all_split_series(splits) == set(series_mapping)
    for split_mapping in splits.values():
        for series_name, items in split_mapping.items():
            assert items == series_mapping[series_name]


def test_split_series_mapping_can_include_test_split():
    series_mapping = {str(idx): [idx] for idx in range(10)}

    splits = split_series_mapping(
        series_mapping,
        train_pct=50,
        val_pct=30,
        test_pct=20,
        seed=3,
    )

    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 5
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 2
    assert _all_split_series(splits) == set(series_mapping)


def test_split_series_mapping_rejects_percentages_not_summing_to_100():
    with pytest.raises(ValueError, match="sum to 100"):
        split_series_mapping({"a": [1]}, train_pct=70, val_pct=20)


def test_load_series_json_validates_shape(tmp_path: Path):
    path = tmp_path / "series.json"
    path.write_text(json.dumps({"a": "not-a-list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must map to a list"):
        load_series_json(path)


def test_write_series_splits_writes_expected_files(tmp_path: Path):
    series_path = tmp_path / "series.json"
    out_dir = tmp_path / "splits"
    series_path.write_text(
        json.dumps({"a": [1], "b": [2], "c": [3], "d": [4]}),
        encoding="utf-8",
    )

    write_series_splits(
        series_json=series_path,
        out_dir=out_dir,
        train_pct=50,
        val_pct=25,
        test_pct=25,
        seed=11,
    )

    assert (out_dir / "train_series.json").is_file()
    assert (out_dir / "val_series.json").is_file()
    assert (out_dir / "test_series.json").is_file()
    train = json.loads((out_dir / "train_series.json").read_text())
    val = json.loads((out_dir / "val_series.json").read_text())
    test = json.loads((out_dir / "test_series.json").read_text())
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1


def test_main_entry_point(tmp_path: Path):
    series_path = tmp_path / "series.json"
    out_dir = tmp_path / "splits"
    series_path.write_text(
        json.dumps({"a": [1], "b": [2], "c": [3], "d": [4]}),
        encoding="utf-8",
    )

    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            "script",
            "--series-json",
            str(series_path),
            "--out-dir",
            str(out_dir),
            "--train-pct",
            "75",
            "--val-pct",
            "25",
            "--seed",
            "13",
        ]
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.argv = original_argv

    assert (out_dir / "train_series.json").is_file()
    assert (out_dir / "val_series.json").is_file()
    assert not (out_dir / "test_series.json").exists()


def _all_split_series(splits):
    result = set()
    for split_mapping in splits.values():
        result.update(split_mapping)
    return result
