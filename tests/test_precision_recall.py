import pytest

from metrics.precision_recall import (
    dataset_precision_recall_at_k,
    precision,
    recall,
    select_k_values,
    select_k_values_log,
    series_precision_recall_at_k,
)


def test_precision_and_recall_basic():
    preds = [1, 2, 3]
    positives = {2, 3, 4}
    assert precision(preds, positives) == pytest.approx(2 / 3)
    assert recall(preds, positives) == pytest.approx(2 / 3)


def test_series_precision_recall_at_k_curve():
    # Series with 3 items; each query has 3 predictions (e.g. N-1 in a 4-item universe).
    series = [0, 1, 2]
    preds_by_query = {
        # Include self to ensure the implementation removes it before slicing top-k.
        0: [0, 1, 2, 3],
        1: [1, 0, 2, 3],
        2: [2, 0, 1, 3],
    }
    curve = series_precision_recall_at_k(preds_by_query, series, max_predictions=2)

    # k=1: 3 hits total, 3 predictions considered, 6 positives considered (2 per query).
    assert curve[0].k == 1
    assert curve[0].precision.value == pytest.approx(1.0)
    assert curve[0].precision.hits == 3
    assert curve[0].precision.denom == 3
    assert curve[0].recall.value == pytest.approx(0.5)
    assert curve[0].recall.hits == 3
    assert curve[0].recall.denom == 6

    # k=2: 6 hits total, 6 predictions considered, 6 positives considered.
    assert curve[1].k == 2
    assert curve[1].precision.value == pytest.approx(1.0)
    assert curve[1].precision.hits == 6
    assert curve[1].precision.denom == 6
    assert curve[1].recall.value == pytest.approx(1.0)
    assert curve[1].recall.hits == 6
    assert curve[1].recall.denom == 6


def test_series_precision_recall_at_k_raises_when_k_too_large():
    preds_by_query = {0: [0, 1, 2], 1: [1, 0, 2]}
    with pytest.raises(ValueError, match="max_predictions"):
        series_precision_recall_at_k(preds_by_query, [0, 1], max_predictions=3)


def test_dataset_precision_recall_at_k_micro_aggregation():
    preds_by_query = {
        0: [0, 1, 2, 3],
        1: [1, 0, 2, 3],
        2: [2, 0, 1, 3],
        3: [3, 0, 1, 2],
    }
    series_to_indices = {"A": [0, 1], "B": [2, 3]}
    result = dataset_precision_recall_at_k(preds_by_query, series_to_indices, max_predictions=1)

    # At k=1, each query retrieves its series mate first, so total_hits=4.
    # precision denom = 1 prediction per query * 4 queries = 4
    # recall denom = 1 positive per query * 4 queries = 4
    assert result.dataset_curve[0].precision.value == pytest.approx(1.0)
    assert result.dataset_curve[0].precision.hits == 4
    assert result.dataset_curve[0].precision.denom == 4
    assert result.dataset_curve[0].recall.value == pytest.approx(1.0)
    assert result.dataset_curve[0].recall.hits == 4
    assert result.dataset_curve[0].recall.denom == 4
    assert set(result.series_curves.keys()) == {"A", "B"}


def test_select_k_values_steps_includes_max_predictions():
    assert select_k_values(max_predictions=7, steps=2) == [1, 3, 5, 7]
    assert select_k_values(max_predictions=8, steps=2) == [1, 3, 5, 7, 8]


def test_select_k_values_log_includes_endpoints_and_is_sorted_unique():
    ks = select_k_values_log(max_predictions=100, num_k=10)
    assert ks[0] == 1
    assert ks[-1] == 100
    assert ks == sorted(ks)
    assert len(ks) == len(set(ks))


