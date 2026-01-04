### Use package rag_info_extractor

Install: pip install -e ".[dev]"  

### Other things to note
- For huggingface models, download them locally and then their model_name/model should be path to snapshots/ref_id file in .cache/huggingface/hub/model/snapshots/hashfolder
- To extract any new information, we need to define its schema in src/rag_info_extractor/info_schema/schemas like other infos in that dir.

# ------------------------------------------------------------- XXX -------------------------------------------------------------


# Instruction on how to create a local package to be used to import in other modules

## Summary:
1. Make `src/rag_app/` the top-level package.
2. Add a `pyproject.toml` so the package can be installed editable with `pip install -e .`.
3. Update `scripts/` to use absolute imports like `from rag_app.document_ingestion.load_docs import load_documents`.
4. Create a dev environment, install the package in editable mode, run scripts and tests.

---

### 1) Move/rename directories (if needed)

Your current `src/` should become a package root containing `rag_app`. If you already have `src/document_ingestion/...`, move them under `src/rag_app/`:

Desired structure:

```
rag-project/
├── pyproject.toml
├── src/
│   └── rag_app/
│       ├── __init__.py
│       ├── document_ingestion/
│       │   ├── __init__.py
│       │   ├── load_docs.py
│       │   └── custom_chunking.py
│       ├── rag_pipeline/
│       ├── information_extractor/
│       └── embeddings_manager.py
├── scripts/
│   ├── ingest_documents.py
│   ├── rag_pipeline.py
│   └── extract_info_json.py
├── tests/
└── ...
```

Notes:

* Keep `scripts/` outside `src/` (good for tooling).
* Ensure each folder under `src/rag_app` has an `__init__.py` (even if empty) so imports are packages.

---

### 2) Add `pyproject.toml` (minimal)

Create `pyproject.toml` in repo root:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-app"
version = "0.0.0"
description = "RAG information extractor"
authors = [{name="Your Name", email="you@unipd.it"}]
requires-python = ">=3.10"

[project.optional-dependencies]
dev = ["pytest", "black", "ruff", "isort", "pre-commit"]
```

---

### 3) Create or update `src/rag_app/__init__.py`

You can keep it simple:

```python
# src/rag_app/__init__.py
__version__ = "0.0.0"
```

---

### 4) Update your scripts to use absolute imports

Change `scripts/` files to import from `rag_app` package. Examples:

#### `scripts/ingest_documents.py`

```python
# scripts/ingest_documents.py
from rag_app.document_ingestion.load_docs import load_documents
from rag_app.document_ingestion.custom_chunking import chunk_documents
from pathlib import Path
```

#### `scripts/rag_pipeline.py`

```python
# scripts/rag_pipeline.py
from rag_app.rag_pipeline.retriever import Retriever
from rag_app.rag_pipeline.generator import Generator
```

#### `scripts/extract_info_json.py`

```python
# scripts/extract_info_json.py
from pathlib import Path
from rag_app.information_extractor.utils import extract_information
from rag_app.document_ingestion.load_docs import load_documents
import json
```

Notes:

* Use `Path` objects for file paths (more robust).
* Keep scripts as thin wrappers that call library code (best practice).

---

### 5) Create a virtual environment and install editable package

From repo root:

```bash
python -m venv .venv           # create venv (or use conda)
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install -e ".[dev]"        # install package in editable mode + dev deps
```
This makes `rag_app.*` available to import from anywhere in your environment.


### 6) Handling config files & package resources

Avoid assuming `cwd` = repo root. Use `pathlib` and package-aware paths:

If `config.yaml` sits in repo root and you need it from package code, either:

* Pass the path into functions from scripts (recommended), or
* Resolve it relative to the script with `Path(__file__).resolve().parents[...]`.

Example (recommended pattern):

```python
# scripts/ingest_documents.py
from pathlib import Path
from rag_app.config import load_config

cfg = load_config(Path("config.yaml"))   # explicit, testable
```

And config loader:

```python
# src/rag_app/config.py
from pathlib import Path
import yaml
from pydantic import BaseSettings

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

This keeps code portable and testable.

---

### 7) Add console scripts (optional, handy)

If you want short CLI commands instead of calling `python scripts/...`, add an entry in `pyproject.toml`:

```toml
[project.scripts]
rag-ingest = "scripts.ingest_documents:main"
rag-run = "scripts.rag_pipeline:run_pipeline"
rag-extract = "scripts.extract_info_json:main"
```

After `pip install -e .` you can run `rag-ingest` from anywhere.

(Alternatively, move CLI entry points to `src/rag_app/cli.py` and point scripts to that; I can show either.)

---

### Extra tips / common gotchas

* If you see `ModuleNotFoundError: No module named 'rag_app'`, check you installed the package in the active venv (`pip show rag-app`) and that you activated the venv.
* Prefer **absolute imports** inside package modules (i.e., `from rag_app.document_ingestion.load_docs import X`) rather than relative imports across top-level packages.
* Use `pip install -e .` whenever you add new modules so the editable install stays current. (If you edit code, it’s immediately effective; re-install only needed if `pyproject.toml`/metadata changed.)
* Keep test fixtures small and committed under `tests/fixtures/` so CI runs fast.
* Add `.env.sample` and a `.gitignore` entry for the venv and data directories.

---


# ------------------------------------------------------------- XXX -------------------------------------------------------------

# RUNNING MODELS LOCALLY:
### Ollama:

### HF (reranker/embedding models)
1. Create a separate directory to save huggingface models (preferebly in users/)
2. Define an env. variable called "HF_MODELS" which stores the absolute path of the directory created in step 1.
3. Copy the HF models (directories) located in the .cache/huggingface/hub/ in the directory created in step 1. (with the same name)


# TEST FILES:
For running tests, first save the outputs in output_to_test.json for the test you want to run.