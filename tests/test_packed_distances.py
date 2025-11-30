import numpy as np
import pytest
import torch

from clip_image_similarity.packed_distances import (
    PackedDistances,
    _infer_n_from_length,
    flatten_upper_triangle,
)


def _build_dist_matrix():
    # Symmetric matrix with distinctive upper-triangular values
    return torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_flatten_upper_triangle_order():
    dist_matrix = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]], dtype=torch.float32
    )
    flat = flatten_upper_triangle(dist_matrix)
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert torch.allclose(flat, expected)


def test_flatten_upper_triangle_requires_square():
    dist_matrix = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]], dtype=torch.float32)
    with pytest.raises(ValueError):
        flatten_upper_triangle(dist_matrix)


@pytest.fixture()
def packed():
    flat = flatten_upper_triangle(_build_dist_matrix()).numpy()
    return PackedDistances(flat)


@pytest.mark.parametrize(
    ("i", "j", "expected"),
    [
        (0, 1, 1.0),
        (0, 3, 3.0),
        (2, 3, 6.0),
        (3, 1, 5.0),  # reversed order should still work
    ],
)
def test_packed_distance_lookup(packed, i, j, expected):
    assert packed.distance(i, j) == pytest.approx(expected)


def test_packed_distance_self_lookup_rejected(packed):
    with pytest.raises(ValueError):
        packed.distance(1, 1)


@pytest.mark.parametrize(
    "invalid_flat",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # not 1D
        np.array([1.0, 2.0], dtype=np.float64),  # unsupported dtype
        np.array([1, 2, 3], dtype=np.int32),  # unsupported dtype
    ],
)
def test_packed_distances_reject_invalid_arrays(invalid_flat):
    with pytest.raises(ValueError):
        PackedDistances(invalid_flat)


def test_packed_distances_save_and_load_round_trip(tmp_path):
    dist_matrix = _build_dist_matrix()
    flat = flatten_upper_triangle(dist_matrix).numpy()
    out = tmp_path / "dists.npz"
    PackedDistances.save(flat, out)
    loaded = PackedDistances.load(out)
    assert loaded.n == dist_matrix.shape[0]
    assert loaded.distance(0, 2) == pytest.approx(2.0)
    assert loaded.distance(1, 3) == pytest.approx(5.0)


def test_packed_distances_load_missing_key_raises(tmp_path):
    out = tmp_path / "bad.npz"
    np.savez_compressed(out, other=np.array([1.0], dtype=np.float32))
    with pytest.raises(KeyError):
        PackedDistances.load(out)


@pytest.mark.parametrize(
    "length",
    [0, -1],
)
def test_infer_n_rejects_non_positive_length(length):
    with pytest.raises(ValueError):
        _infer_n_from_length(length)


def test_infer_n_rejects_incompatible_length():
    with pytest.raises(ValueError):
        _infer_n_from_length(5)  # 5 is not n*(n-1)/2 for any integer n
