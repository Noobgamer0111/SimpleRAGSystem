import numpy as np
import importlib.util
from pathlib import Path

# Import the RAGScript module by file path so tests can run when the package isn't installed
proj_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("RAGScript", str(proj_root / "RAGScript.py"))
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_cosine_similarity_perfect_and_orthogonal():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])

    # identical vectors -> similarity 1.0
    assert abs(rs.cosine_similarity(v1, v2) - 1.0) < 1e-6

    # orthogonal vectors -> similarity close to 0.0
    assert abs(rs.cosine_similarity(v1, v3)) < 1e-6


class DummyEmbedModel:
    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, text):
        # ignore text, return predefined vector
        return self.vec


def test_retrieve_ranking():
    # create three docs with embeddings
    docs = [
        ("doc1", np.array([1.0, 0.0])),
        ("doc2", np.array([0.0, 1.0])),
        ("doc3", np.array([0.7, 0.7])),
    ]

    # query vector points in +x direction so doc1 is best
    dummy = DummyEmbedModel([1.0, 0.0])
    results = rs.retrieve("any question", docs, dummy, top_n=3)

    # results are (doc_text, similarity)
    assert results[0][0] == "doc1"
    assert results[1][0] in ("doc3", "doc2")
