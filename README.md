# SimpleRAGSystem

Setup and run instructions

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
```

2. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Run unit tests with pytest:

```bash
pytest -q
```

4. Run the script (LM Studio must be installed and models available):

```bash
python RAGScript.py --file-path cat-facts.txt --top-n 3
```

Notes:
- `requirements.txt` will install `lmstudio`, `requests`, `numpy`, and `pytest`.
- `lmstudio` requires local models to be downloaded using the `lms get` CLI or LM Studio app as described in LM Studio docs.
