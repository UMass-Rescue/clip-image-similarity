import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "collapse_subseries_json.py"
)
SPEC = importlib.util.spec_from_file_location("collapse_subseries_json", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

collapse_subseries_mapping = MODULE.collapse_subseries_mapping
load_series_mapping = MODULE.load_series_mapping
load_series_to_subseries_mapping = MODULE.load_series_to_subseries_mapping
write_collapsed_series_mapping = MODULE.write_collapsed_series_mapping


def test_collapse_subseries_mapping_merges_multiple_subseries():
    flat = {
        "A_subseries_0": ["a0.jpg", "a1.jpg"],
        "A_subseries_1": ["a2.jpg"],
        "B": ["b0.jpg"],
    }
    nested = {
        "A": {"A_subseries_0": [0, 1], "A_subseries_1": [2]},
        "B": {"B": [3]},
    }

    collapsed = collapse_subseries_mapping(
        series_mapping=flat,
        series_to_subseries=nested,
    )

    assert collapsed == {
        "A": ["a0.jpg", "a1.jpg", "a2.jpg"],
        "B": ["b0.jpg"],
    }


def test_collapse_subseries_mapping_allows_split_subset():
    flat = {"A_subseries_1": [2], "C": [5]}
    nested = {
        "A": {"A_subseries_0": [0, 1], "A_subseries_1": [2]},
        "B": {"B": [3, 4]},
        "C": {"C": [5]},
    }

    collapsed = collapse_subseries_mapping(
        series_mapping=flat,
        series_to_subseries=nested,
    )

    assert collapsed == {"A": [2], "C": [5]}


def test_collapse_subseries_mapping_dedupes_values_within_original_series():
    flat = {
        "A_subseries_0": [1, 2],
        "A_subseries_1": [2, 3],
    }
    nested = {"A": {"A_subseries_0": [1, 2], "A_subseries_1": [2, 3]}}

    collapsed = collapse_subseries_mapping(
        series_mapping=flat,
        series_to_subseries=nested,
    )

    assert collapsed == {"A": [1, 2, 3]}


def test_collapse_subseries_mapping_rejects_unknown_flat_label():
    with pytest.raises(ValueError, match="not found"):
        collapse_subseries_mapping(
            series_mapping={"unknown": [1]},
            series_to_subseries={"A": {"A_subseries_0": [1]}},
        )


def test_load_series_mapping_validates_shape(tmp_path: Path):
    path = tmp_path / "series.json"
    path.write_text(json.dumps({"A": "not-a-list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must map to a list"):
        load_series_mapping(path)


def test_load_series_to_subseries_mapping_validates_shape(tmp_path: Path):
    path = tmp_path / "series_to_subseries.json"
    path.write_text(json.dumps({"A": ["not-an-object"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="must map to an object"):
        load_series_to_subseries_mapping(path)


def test_write_collapsed_series_mapping_writes_parent_dirs(tmp_path: Path):
    flat_path = tmp_path / "flat.json"
    nested_path = tmp_path / "nested.json"
    output_path = tmp_path / "outputs" / "collapsed.json"
    flat_path.write_text(
        json.dumps({"A_subseries_0": ["a0.jpg"], "A_subseries_1": ["a1.jpg"]}),
        encoding="utf-8",
    )
    nested_path.write_text(
        json.dumps({"A": {"A_subseries_0": [0], "A_subseries_1": [1]}}),
        encoding="utf-8",
    )

    write_collapsed_series_mapping(
        series_json=flat_path,
        series_to_subseries_json=nested_path,
        output_json=output_path,
    )

    assert json.loads(output_path.read_text()) == {"A": ["a0.jpg", "a1.jpg"]}


def test_main_entry_point(tmp_path: Path):
    flat_path = tmp_path / "flat.json"
    nested_path = tmp_path / "nested.json"
    output_path = tmp_path / "collapsed.json"
    flat_path.write_text(json.dumps({"A_subseries_0": [1]}), encoding="utf-8")
    nested_path.write_text(
        json.dumps({"A": {"A_subseries_0": [1], "A_subseries_1": [2]}}),
        encoding="utf-8",
    )

    original_argv = sys.argv.copy()
    try:
        sys.argv = [
            "script",
            "--series-json",
            str(flat_path),
            "--series-to-subseries-json",
            str(nested_path),
            "--output-json",
            str(output_path),
        ]
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.argv = original_argv

    assert json.loads(output_path.read_text()) == {"A": [1]}
