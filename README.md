# medicoder-rag

A retrieval-augmented generation (RAG) system that assigns **ICD-10-CM medical codes**
to free-text clinical notes. Given a medical note, it retrieves candidate codes from
the full billable ICD-10-CM codebook (74,719 codes) and uses a local LLM to select the
codes that apply — running entirely offline on consumer hardware.

Evaluated on 54 clinical cases: **0.532 recall, 0.152 precision** — 84 of 158
ground-truth codes recovered — at ~15s/case.

---

## Problem

Assigning ICD-10-CM codes to clinical notes is hard for three reasons:

- **Multiple diagnoses per note.** A single case describes several distinct conditions.
- **Vernacular gap.** Clinicians write "herniated disc"; the codebook says
  "intervertebral disc displacement".
- **A huge, near-duplicate search space.** 74,719 billable codes, many differing only
  in encounter type or severity.

## Architecture

Two phases: an **offline** index build (run once) and an **online** inference pipeline
(per case).

### Offline — build the vector index (once)

```
icd10cm_order_2026.txt (98,186 lines)
        │  parse_codebook.py   — keep billable (flag-1) codes, drop parent headers
        ▼
icd10_billable_codes.jsonl (74,719 codes)
        │  embed_codebook.py   — S-PubMedBert-MS-MARCO
        ▼
embeddings.npy (74,719 vectors)
        │  index_codebook.py   — load into ChromaDB (cosine, HNSW)
        ▼
chroma_db/
```

### Online — per case (`fiftyfour_cases.py`)

```
Medical note (free text)
        │  LLM CALL 1 (qwen3:8b) — extract distinct conditions
        ▼
Individual conditions (~10–15 per case)
        │  embed each (S-PubMedBert, batched) → retrieve top-15 per condition
        ▼
Candidate codes (~130)
        │  merge + dedup (keep min distance) + sort + cap at 40   [deterministic, no LLM]
        ▼
≤40 candidates
        │  LLM CALL 2 (qwen3:8b) — select final codes from note + candidates
        ▼
Predicted ICD-10 codes → evaluate (recall, precision, time, tokens)
```

**The critical insight — query decomposition.** Embedding the *whole note* as one
vector gave ≈0 recall: a diagnostic showed only **28/158** ground-truth codes reached
even the top-100. The root cause was embedding dilution — a multi-diagnosis note
averages into a vector close to no single code. Splitting the note into per-condition
queries (LLM call 1) and retrieving for each is what makes retrieval work.

## Key design decisions

- **Embedding model — S-PubMedBert-MS-MARCO.** PubMedBERT is pretrained on biomedical
  text and MS-MARCO fine-tuned for retrieval. (Caveat: trained on journal prose, not
  ICD billing labels — the vernacular gap isn't fully closed.)
- **LLM — qwen3:8b via Ollama.** Local, offline, open-source. 8B fits consumer hardware
  (developed on a 16GB MacBook M4: ~5GB model weights + ~400MB embeddings + ~1GB
  ChromaDB).
- **Deterministic middle (dedup/sort/cap).** No LLM in the merge step — repeatable and
  cheap.
- **Temperature 0.0** — deterministic output, so Version A/B comparisons are fair.
- **`num_predict=100`** — a safety brake. Under greedy decoding the model once fell into
  an infinite loop repeating codes and hung the run; a hard token cap stops that.
- **Cap candidates at 40** — passing all ~130 made the LLM emit markdown essays instead
  of code lists; 40 keeps it focused.

## Results

### Version comparison (does the second LLM earn its place?)

| Metric (per-case mean) | Version A (retrieval only) | Version B (retrieval + LLM) |
|---|---|---|
| Avg recall | 0.308 | **0.609** |
| Avg precision | 0.048 | **0.237** |
| Avg time/case | 6.06s | 15.11s |

Version B nearly doubles recall and quintuples precision at ~2.5× the latency — so the
selection LLM stays.

These two columns are per-case means, not the overall counts reported below: Version A
was measured on a partial run, so the two versions are only comparable to each other
under the same averaging. Re-run both over all 54 cases to compare them on overall
recall and precision.

### Final — all 54 cases

Metrics are counted across all cases (total correct ÷ total codes), not averaged
per case — a per-case mean weights a 1-code case the same as a 10-code case, which
measures the typical case rather than the pipeline.

| Metric | Value |
|---|---|
| Overall recall | **0.532** (84 of 158 ground-truth codes found) |
| Overall precision | **0.152** (84 of 551 predicted codes correct) |
| Retrieval recall | 0.671 (106/158 ground-truth codes reached the pool) |
| LLM retention | 79% (84 of the 106 pooled codes selected) |
| Avg time/case | 15.11s |
| Total tokens | 70,751 (67,443 prompt + 3,308 completion) |

For reference, the per-case means are 0.609 recall and 0.237 precision — higher than
the overall figures because cases with few ground-truth codes are easier to score
well on and carry equal weight in that average.

Read together: retrieval is the bottleneck (52 of 158 codes, 33%, never reach the
pool), while the LLM keeps 79% of what it *does* see — which is why the final-stage
LLM is worth keeping. See `data/results_full.json` for per-case detail.

## Setup

```bash
# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Python dependencies
pip install -r requirements.txt

# 3. Local LLM (extraction + selection)
ollama pull qwen3:8b        # requires Ollama: https://ollama.com
```

Requirements: **Python 3.11+** and **Ollama** running locally
(`http://localhost:11434`). The embedding model
(`pritamdeka/S-PubMedBert-MS-MARCO`) downloads automatically on first use.

## Data

The large/proprietary data files are **not committed** (see `.gitignore`):

- **ICD-10-CM codebook** (`data/icd10cm_order_2026.txt`, ~14MB, public domain) —
  download from the CMS ICD-10-CM files page (search "CMS ICD-10-CM 2026"): grab the
  *"Code Descriptions in Tabular Order"* archive and place `icd10cm_order_2026.txt`
  in `data/`.
- **Evaluation cases** (`data/icd10_cm_cases.json`) — the original assignment dataset is
  not redistributed here. A small **synthetic** `data/sample_cases.json` (same schema)
  is provided so you can run the pipeline end-to-end. Supply your own cases in the same
  format to evaluate on real data.

Cases file schema:

```json
{ "cases": [ { "medical_note": "…", "icd10_cm": { "codes": ["N30.00"] } } ] }
```

## Running the pipeline (in order)

From the project root with the venv active. Steps 1–3 build the index and only need to
run once (or when the codebook changes).

```bash
python parse_codebook.py     # 1. codebook -> data/icd10_billable_codes.jsonl
python embed_codebook.py     # 2. embed    -> data/embeddings.npy
python index_codebook.py     # 3. index    -> chroma_db/
python retrieve_check.py     # (optional) sanity-check retrieval
python fiftyfour_cases.py    # 4. run + evaluate -> data/results_full.json
```

**Quick demo without the real dataset:** point the pipeline at the synthetic sample by
setting `CASES_PATH = "data/sample_cases.json"` at the top of `fiftyfour_cases.py`
(and `MAX_CASES = None`).

## Configuration

Knobs at the top of `fiftyfour_cases.py`:

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | LLM for extraction + selection |
| `EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | query/codebook embeddings |
| `CANDIDATES_PER_CONDITION` | `15` | codes retrieved per extracted condition |
| `MAX_CANDIDATES_FOR_LLM` | `40` | cap on the merged candidate pool |
| `MAX_CASES` | `None` | limit cases processed (`None` = all) |

## Project structure

```
.
├── parse_codebook.py     # parse ICD-10-CM order file -> billable-codes JSONL
├── embed_codebook.py     # embed code descriptions -> .npy
├── index_codebook.py     # load vectors into ChromaDB
├── retrieve_check.py     # retrieval helpers + standalone sanity check
├── fiftyfour_cases.py    # main pipeline (decompose -> retrieve -> select -> eval)
├── requirements.txt
├── data/
│   ├── sample_cases.json     # synthetic demo cases (committed)
│   └── results_full.json     # evaluation output (committed as evidence)
└── docs/
    └── section2_design.pdf   # full design write-up
```

## Limitations

- **Retrieval ceiling of 67.1%** — 52/158 ground-truth codes never reach the pool,
  so no amount of LLM improvement can recover them.
- Embedding model trained on PubMed prose, not ICD billing terminology.
- No hybrid (lexical + semantic) search.
- Low precision (0.152): ~10 codes predicted per case, ~1.6 correct.
- Condition extraction is left to the LLM; merged/under-split conditions go unchecked
  and dilute their embeddings.

## Future improvements

- **Codebook enrichment** — augment each code with official CMS/NCHS synonyms and
  inclusion terms before embedding, closing the vernacular gap. (Authoritative data,
  *not* LLM-invented.)
- **Hybrid retrieval** — combine embeddings with BM25 for exact-term matches.
- **Few-shot prompting** — show worked examples to guide selection on ambiguous cases.
- **Structured JSON output** — force code-list output, eliminating essay/`<think>` noise.
- **Self-consistency** — run selection several times and keep majority-voted codes.
- **Hallucination guardrail** — drop any predicted code not in the candidate pool.
- **Encounter-type consistency** — detect contradictory A/D/S suffixes for the same
  condition (e.g. `T18128A` vs `T18128D`) and resolve to one.

---

*AI-assisted: scaffolding generated with Claude, reviewed and understood by the author.*
