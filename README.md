# Rag Information Extractor

Rag Information Extractor is a RAG (Retrieval-Augmented Generation) pipeline that extracts structured information from Italian company documents (statuti / financial statements) and evaluates the results. 
This repository is a reconstructed version of my internship project.
To respect proprietary confidentiality and maintain a manageable project scope, this reproduction omits certain  features and internal dependencies present in the original.

Main components:
- Document ingestion (PDF -> chunks -> vector store)
- Retrieval -> optional pruning
- Cross-encoder re-ranking
- LLM-based structured extraction (returns valid JSON)
- Evaluation utilities (matching + accuracy/runtime summaries)

## Repository layout (high level)
- `scripts/` : pipeline execution entrypoints (ingestion + extraction)
- `apis/` : local microservices for embeddings, pruning, and re-ranking
- `src/rag_info_extractor/` : RAG pipeline code (LangGraph nodes, schemas, utilities)
- `evaluations/` : evaluation and matching scripts

## Prerequisites
- Python `>= 3.10`
- A working LLM runtime:
  - Local via Ollama (default option in this repo)
  - Optionally Google GenAI (see `USE_GOOGLE_API` in `config.yaml`)
- (Recommended) Local microservices:
  - Embedding service
  - Pruner service
  - Re-ranker service

## Install
1. Create/activate a virtual environment (the repo scripts expect `.venv`).
2. Install dependencies:
   - `pip install -r requirements.txt`
   - or `pip install -e .`

## Configuration
Edit `config.yaml`:
- `BASE_DIR`: absolute path to this project root
- `OLLAMA_HOST`, `LLM_MODEL`, `EVALUATOR_LLM`, `EXTRACTOR_LLM`
- `EMBEDDING_MODEL_NAME`, `RERANKER_MODEL`, `PRUNER_MODEL`
- `CHUNKS_TYPE`: ingestion chunking mode
- `RAG_PIPELINE`: pipeline nodes to run (example: `["retrieve", "cross_encode_rerank", "generate"]`)

### Environment files
The microservices and the connector code load environment variables from dotenv files in `BASE_DIR`.

Create at least:
- `BASE_DIR/.env`

Additionally, note that:
- `apis/pruner_api.py` and `apis/re_ranker_api.py` load `BASE_DIR/.env.txt`
- `src/rag_info_extractor/utils/apis_connector.py` loads `BASE_DIR/.env`

So, depending on your setup, you may need to create both `.env` and `.env.txt` (or ensure the required variables exist in both).

Expected `.env` variables (used by the connector):
- `RERANKER_API`: URL of the re-ranker microservice endpoint (POST)
- `PRUNER_API`: URL of the pruner microservice endpoint (POST)
- `EMBEDDING_API`: URL of the embedding microservice endpoint (POST)

Expected microservice model variables:
- The microservices infer a local model path from environment variables derived from the model names.
  - Example (re-ranker): it builds an env var key from `RERANKER_MODEL` and looks it up with `os.environ.get(...)`.

## Microservices (local)
The microservices are implemented with FastAPI and started by `run_apis.sh`.

`run_apis.sh` starts (in background):
- `apis/embedding_api.py`
- `apis/pruner_api.py`
- `apis/re_ranker_api.py`

Their endpoints are:
- Re-ranker: `POST /rerank` (default port `8000`)
- Pruner: `POST /prune` (default port `8001`)
- Embedding: `POST /embed` (default port `8002`)

Make sure your `.env` sets:
- `RERANKER_API` to something like `http://localhost:8000/rerank`
- `PRUNER_API` to something like `http://localhost:8001/prune`
- `EMBEDDING_API` to something like `http://localhost:8002/embed`

## Running the pipeline
There are two main phases:
1. Document ingestion (PDF -> chunks -> store)
2. Extraction (RAG -> JSON output)

### 1) Doc ingestion
Run:
```bash
./run_doc_ingestion.sh
```

Notes:
- The script starts Ollama (`ollama serve`) and then runs `scripts/ingest_docs.py`.
- On Windows, use a shell that can run `.sh` scripts (e.g., Git Bash, WSL).

### 2) Extraction
Run:
```bash
./run_extraction.sh
```

Notes:
- `run_extraction.sh` starts Ollama and then runs `scripts/extract_info.py`.
- `extract_info.py` is responsible for orchestrating the RAG pipeline + structured JSON extraction.

### Optional: start only microservices
If you only want the embedding/pruner/reranker services:
```bash
./run_apis.sh
```

## Data formats
This repo expects “combined” JSON files and produces prediction JSONs.

### Input: `combined_data.json`
The “combined raw” JSON is expected to contain these keys:
- `values`
- `raw_contexts`
- `raw_contexts_ids`
- `raw_qa`

Raw JSONs are typically organized like:
```json
{
  "Azienda_name": {
    "BILANCI_E_UTILI": { "values": { "...": "" }, "raw_contexts": { "...": "" }, "...": "..." },
    "COMPENSO_DEGLI_AMMINISTRATORI": { "...": "..." }
  }
}
```

### Output: `pred.json`
Predictions are typically saved under:
`run/<TRAIN_OR_TEST>/run_time/pred.json`

The structure mirrors the raw groups and includes (example categories):
- `output`: the extracted structured JSON
- `retrieved_docs` / `re_ranked_docs`: chunk ids/text used by the pipeline
- `rag_qa`: generated Q/A for the QA-style part (if enabled)
- `run_times`: timings per module

## Evaluation
Evaluation scripts live under `evaluations/`.

Typical workflow:
1. Ensure raw chunk ids exist (run):
   - `python evaluations/utils/find_raw_chunks_ids.py`
2. Run extraction (see `Running the pipeline`)
3. Build match files:
   - `python evaluations/utils/match_pred_output_llm.py`
   - `python evaluations/utils/match_pred_rag_QA_llm.py`
4. Compute summaries:
   - `python evaluations/eval_generation.py`
   - `python evaluations/eval_overall.py`
   - `python evaluations/aggregate_eval_overall.py`

## Notes / known TODOs (from existing docs)
- Verify/iterate on:
  - query optimization strategies in `analyze_query`
  - extraction LLM choice and prompt robustness
  - evaluation metrics for faithfulness / answer relevance / conciseness
- There is an ongoing TODO to integrate the extraction process in an async and/or frontend-friendly way.

