import numpy as np
import pytest
import torch

from clip_image_similarity.topk import (
    TopKNeighbors,
    extract_topk_neighbors,
    save_topk_neighbors,
)


def small_dist_matrix():
    return torch.tensor(
        [[0.0, 3.0, 1.0], [3.0, 0.0, 2.0], [1.0, 2.0, 0.0]], dtype=torch.float32
    )


def test_extract_topk_neighbors_float32():
    dist = small_dist_matrix()
    indices, distances = extract_topk_neighbors(dist.clone(), top_k=1, dtype="float32")
    # For each row, the nearest neighbor index and distance
    assert indices.shape == (3, 1)
    assert distances.shape == (3, 1)
    assert indices[:, 0].tolist() == [2, 2, 0]
    assert distances[:, 0].tolist() == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(1.0)]
    assert distances.dtype == np.float32


def test_extract_topk_neighbors_float16():
    dist = small_dist_matrix()
    _, distances = extract_topk_neighbors(dist.clone(), top_k=1, dtype="float16")
    assert distances.dtype == np.float16


@pytest.mark.parametrize("top_k", [0, -1, 3])
def test_extract_topk_neighbors_invalid_topk(top_k):
    dist = small_dist_matrix()
    with pytest.raises(ValueError):
        extract_topk_neighbors(dist, top_k=top_k)


def test_extract_topk_neighbors_requires_square():
    dist = torch.ones((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        extract_topk_neighbors(dist, top_k=1)


def test_topkneighbors_neighbors_and_distance(tmp_path):
    dist = small_dist_matrix()
    idx, dists = extract_topk_neighbors(dist.clone(), top_k=2, dtype="float32")
    out = tmp_path / "topk.npz"
    save_topk_neighbors(out, indices=idx, distances=dists, dtype="float32")
    neighbors = TopKNeighbors.load(out)

    assert neighbors.n == 3
    assert neighbors.k == 2
    assert neighbors.neighbors(0) == [(2, pytest.approx(1.0)), (1, pytest.approx(3.0))]
    assert neighbors.distance_if_present(1, 0) == pytest.approx(3.0)
    assert neighbors.distance_if_present(1, 1) is None  # not stored as neighbor
    assert neighbors.distance_if_present(2, 1) == pytest.approx(2.0)


def test_topkneighbors_neighbors_bounds_check(tmp_path):
    dist = small_dist_matrix()
    idx, dists = extract_topk_neighbors(dist.clone(), top_k=1)
    out = tmp_path / "topk.npz"
    save_topk_neighbors(out, idx, dists, dtype="float32")
    neighbors = TopKNeighbors.load(out)
    with pytest.raises(ValueError):
        neighbors.neighbors(-1)
    with pytest.raises(ValueError):
        neighbors.neighbors(3)
    with pytest.raises(ValueError):
        neighbors.distance_if_present(-1, 0)
    with pytest.raises(ValueError):
        neighbors.distance_if_present(0, -1)


@pytest.mark.parametrize(
    ("indices", "distances"),
    [
        (np.ones((2, 2, 1), dtype=np.uint16), np.ones((2, 2), dtype=np.float32)),  # ndim mismatch
        (np.ones((2, 2), dtype=np.uint16), np.ones((2, 1), dtype=np.float32)),  # shape mismatch
        (np.ones((2, 2), dtype=np.uint16), np.ones((2, 2), dtype=np.int32)),  # bad distances dtype
        (np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)),  # indices not int
    ],
)
def test_topkneighbors_init_validation(indices, distances):
    with pytest.raises(ValueError):
        TopKNeighbors(indices, distances)


def test_topkneighbors_init_topk_mismatch():
    indices = np.ones((2, 2), dtype=np.uint16)
    distances = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        TopKNeighbors(indices, distances, top_k=3)


def test_topkneighbors_load_missing_keys(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, indices=np.ones((1, 1)), distances=np.ones((1, 1)))
    with pytest.raises(KeyError):
        TopKNeighbors.load(bad)


def test_topkneighbors_load_dtype_mismatch(tmp_path):
    path = tmp_path / "mismatch.npz"
    np.savez_compressed(
        path,
        indices=np.ones((1, 1), dtype=np.uint16),
        distances=np.ones((1, 1), dtype=np.float32),
        top_k=1,
        dtype="float16",  # metadata mismatches real float32 distances
    )
    # Force dtype mismatch detection by overriding __init__ to keep true distances dtype
    original_init = TopKNeighbors.__init__

    def tweaked_init(self, indices, distances, *, top_k=None, dtype=None):
        original_init(self, indices, distances, top_k=top_k, dtype=dtype)
        # Override dtype to reflect array dtype, creating mismatch with stored metadata
        self.dtype = str(distances.dtype)

    try:
        TopKNeighbors.__init__ = tweaked_init
        with pytest.raises(ValueError):
            TopKNeighbors.load(path)
    finally:
        TopKNeighbors.__init__ = original_init


def test_topkneighbors_load_index_dtype_mismatch(tmp_path):
    path = tmp_path / "idx_mismatch.npz"
    np.savez_compressed(
        path,
        indices=np.ones((1, 1), dtype=np.uint16),
        distances=np.ones((1, 1), dtype=np.float32),
        top_k=1,
        dtype="float32",
        index_dtype="int32",  # does not match uint16 indices
    )
    with pytest.raises(ValueError):
        TopKNeighbors.load(path)
