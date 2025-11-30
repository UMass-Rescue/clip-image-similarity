import torch

from clip_image_similarity.similarity import SimilarityComputer


def test_similarity_computer_cosine_and_distance(monkeypatch):
    logs = []
    monkeypatch.setattr(
        "clip_image_similarity.similarity.log", lambda msg: logs.append(msg)
    )

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )  # orthogonal vectors
    sim_comp = SimilarityComputer(device="cpu")
    sim = sim_comp.cosine_similarity_matrix(embeddings)

    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(sim, expected)
    dist = sim_comp.similarity_to_distance(sim)
    assert torch.allclose(dist, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert "Computing cosine similarity matrix" in logs[0]
    assert "Similarity matrix computed" in logs[1]
