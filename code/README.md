# MLE Hiring Challenge Agent

This directory contains the support-ticket resolution agent. The primary entry point is `code/main.py`; it reads `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv` with the required evaluator columns.

## Setup

Run these commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

If you are on Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code\requirements.txt
```

## Local Models

The agent uses Ollama when available and falls back to deterministic rules when a local model is unavailable. For the intended run, install Ollama and pull both models:

```bash
ollama pull llama3.2:3b
ollama pull llama3.2:8b
```

The configured model roles are:

- `llama3.2:3b`: routing, safety confirmation, retrieval query planning, evidence judging, and tool-intent labeling
- `llama3.2:8b`: response generation and reflection

The pipeline uses `temperature=0.0` and `seed=42` in `code/config.py`.

## Build The Retrieval Index

Build the hybrid FAISS + BM25 index before running the full agent:

```bash
python code/build_index.py
```

This creates the `index/` artifacts used by `core.hybrid_retriever.HybridRetriever`:

- `index/faiss.index`
- `index/bm25_index.pkl`
- `index/chunks_metadata.pkl`
- `index/embeddings.npy`

The first run may download the Sentence Transformers embedding model `sentence-transformers/all-MiniLM-L6-v2`.

## Reproduce `support_tickets/output.csv`

From the repository root, run:

```bash
python code/build_index.py
python code/main.py
```

`code/main.py` processes every row in `support_tickets/support_tickets.csv`, writes the final CSV to `support_tickets/output.csv`, runs structural validation, and prints final stats including replied/escalated ratio, source coverage, average confidence, runtime, and validator exit code.

## Run Validation Only

```bash
python code/validate_output.py
```

The validator checks structure only: headers, row count, enum values, confidence range, and valid JSON arrays in `actions_taken`. It does not score answer correctness.

## Expected Runtime

Runtime depends heavily on CPU/GPU availability and whether Ollama is warm. On CPU with local models, expect several minutes for all tickets. If Ollama is unavailable, the rule fallbacks run much faster but responses may be less polished.

## Determinism Notes

The code pins deterministic settings where practical:

- `code/config.py` sets `SEED = 42` and `TEMPERATURE = 0.0`.
- `code/build_index.py` seeds Python, NumPy, and Torch before embedding/index construction.
- `code/main.py` seeds Python, NumPy, and Torch at startup.
- Retrieval uses deterministic BM25/FAISS score fusion and sorted reranking.

Small differences can still occur across hardware, model versions, Ollama versions, and first-time embedding downloads.
