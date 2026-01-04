import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

# Import the RAGScript module by file path
proj_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("RAGScript", str(proj_root / "RAGScript.py"))
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


class DummyLLMInstance:
    def __init__(self):
        self.prompt = None

    def respond_stream(self, prompt):
        self.prompt = prompt
        yield SimpleNamespace(content="Answer part 1 ")
        yield SimpleNamespace(content="Answer part 2")

    def respond(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(output_text="Full answer.")


class DummyEmbedInstance:
    def embed_query(self, text):
        # simple deterministic vector from text
        return [len(text), sum(ord(c) for c in text) % 100]


class DummyClient:
    class llm:
        @staticmethod
        def load_new_instance(name, ttl=None):
            return DummyLLMInstance()

    class embedding_model:
        @staticmethod
        def load_new_instance(name, ttl=None):
            return DummyEmbedInstance()


def test_integration_with_mocked_lmstudio(monkeypatch, tmp_path, capsys):
    # prepare a small knowledge base file
    kb = tmp_path / "kb.txt"
    kb.write_text("Cats purr when happy.\nCats sleep a lot.\n")

    # Patch LM Studio client to return our dummy client
    monkeypatch.setattr(rs.lms, "get_default_client", lambda: DummyClient())

    # Patch input() to provide a query and argv for CLI args
    monkeypatch.setattr('builtins.input', lambda prompt='': "Do cats purr?")
    monkeypatch.setattr(sys, "argv", ["RAGScript.py", "--file-path", str(kb), "--top-n", "2"])

    # Run main; it should use the mocked client and print results
    rs.main()

    captured = capsys.readouterr()
    assert "Retrieved Knowledge:" in captured.out
    # ensure at least part of the mocked answer appears
    assert "Answer part 1" in captured.out or "Full answer." in captured.out
