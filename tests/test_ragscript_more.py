import numpy as np
import importlib.util
from pathlib import Path

# Import the RAGScript module by file path
proj_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("RAGScript", str(proj_root / "RAGScript.py"))
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_cosine_similarity_zero_vector():
    v_zero = np.array([0.0, 0.0])
    v = np.array([1.0, 2.0])
    assert rs.cosine_similarity(v_zero, v) == 0.0
    assert rs.cosine_similarity(v, v_zero) == 0.0


class DummyEmbed:
    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, text):
        return self.vec


def test_retrieve_top_n_respects_parameter():
    docs = [
        ("a", np.array([1.0, 0.0])),
        ("b", np.array([0.0, 1.0])),
        ("c", np.array([0.5, 0.5])),
        ("d", np.array([0.9, 0.1])),
    ]
    embed = DummyEmbed([1.0, 0.0])
    results = rs.retrieve("q", docs, embed, top_n=2)
    assert len(results) == 2
    assert results[0][0] == "a"
