# medicoder-rag

A retrieval-augmented (RAG) system that assigns ICD-10-CM medical codes to clinical
notes. Given a free-text medical note, it retrieves candidate codes from the full
billable ICD-10-CM codebook and uses a local LLM to select the codes that apply.

## What it does

The pipeline runs in two offline stages (build the index) and one online stage
(classify cases):

**Index build (run once):**
1. **Parse** the official ICD-10-CM 2026 order file into a clean JSONL of the
   ~74.7k *billable* codes (`parse_codebook.py`).
2. **Embed** each code's description into a vector with S-PubMedBert
   (`embed_codebook.py`).
3. **Index** those vectors into a persistent ChromaDB collection
   (`index_codebook.py`).

**Classification (per case) — `fiftyfour_cases.py`:**
1. **Decompose** the note into distinct conditions (LLM pass 1: qwen3).
2. **Retrieve** the top-K codes per condition from ChromaDB (cosine similarity).
3. **Merge** candidates: dedup by code keeping the closest distance, sort, and cap
   the pool at 40.
4. **Select** the final codes from that pool (LLM pass 2: qwen3).
5. **Evaluate** predictions against ground truth (recall + precision).

This two-stage design (decompose → retrieve → select) exists because embedding a
whole multi-condition note as a single query dilutes retrieval; decomposing into
per-condition queries substantially improves the chance the correct code reaches
the candidate pool.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally (default `http://localhost:11434`)
- The ICD-10-CM 2026 order file at `data/icd10cm_order_2026.txt`
- The evaluation cases at `data/icd10_cm_cases.json`

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull the LLM used for extraction + selection
ollama pull qwen3:8b
```

The embedding model (`pritamdeka/S-PubMedBert-MS-MARCO`) is downloaded
automatically by `sentence-transformers` on first use.

## Running the pipeline (in order)

Run these from the project root with the venv active. Steps 1–3 build the vector
index and only need to be run once (or whenever the codebook changes).

```bash
# 1. Parse the codebook -> data/icd10_billable_codes.jsonl
python parse_codebook.py

# 2. Embed code descriptions -> data/embeddings.npy
python embed_codebook.py

# 3. Index vectors into ChromaDB -> chroma_db/
python index_codebook.py

# (optional) sanity-check retrieval
python retrieve_check.py

# 4. Run the full evaluation over all 54 cases
python fiftyfour_cases.py
```

## Configuration

Key knobs live at the top of `fiftyfour_cases.py`:

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | LLM for extraction + selection |
| `EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | query/codebook embeddings |
| `CANDIDATES_PER_CONDITION` | `15` | codes retrieved per extracted condition |
| `MAX_CANDIDATES_FOR_LLM` | `40` | cap on the merged candidate pool |
| `MAX_CASES` | `None` | limit cases processed (`None` = all 54) |

The LLM calls use `temperature=0.0` (deterministic) and `num_predict=100`.

## Results

The evaluation writes detailed, per-case results to **`data/results_full.json`**,
and prints a summary to the console including:

- **Average recall** and **average precision** across cases
- **Average time per case** and total token usage
- **Two-stage recall breakdown** — retrieval recall (how many ground-truth codes
  reached the candidate pool) vs. final recall (how many the LLM actually kept)

## Files

| File | Role |
|---|---|
| `parse_codebook.py` | Parse ICD-10-CM order file → billable-codes JSONL |
| `embed_codebook.py` | Embed code descriptions → `.npy` |
| `index_codebook.py` | Load vectors into ChromaDB |
| `retrieve_check.py` | Retrieval helpers + standalone sanity check |
| `fiftyfour_cases.py` | Main evaluation pipeline |

---

*AI-assisted: scaffolding generated with Claude, reviewed and understood by author.*
