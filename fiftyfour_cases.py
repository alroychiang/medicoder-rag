# AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""
Main pipeline: process all 54 medical cases through retrieval + LLM classification.

For each case:
  1. Retrieve top-K candidate ICD-10 codes from ChromaDB
  2. Prompt Ollama to select the correct codes from candidates
  3. Compare predictions against ground truth
  4. Report recall, time taken, and token usage

AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""

import json
import time
import requests
from retrieve_check import get_collection
from sentence_transformers import SentenceTransformer

# --- Config ---
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
CANDIDATES_PER_CONDITION = 15  # top-K retrieved per extracted condition
MAX_CANDIDATES_FOR_LLM = 40  # cap the merged pool sent to the selection LLM
MAX_CASES = None  # cap cases processed; set to None to run all 54
CASES_PATH = "data/icd10_cm_cases.json"
RESULTS_PATH = "data/results_full.json"


def load_cases(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["cases"]


def build_prompt(medical_note: str, candidates: list[dict]) -> str:
    """
    Build a prompt that gives the LLM the medical note and candidate codes,
    and asks it to select the correct ICD-10 codes.
    """
    candidate_list = "\n".join(
        f"  {c['code']} - {c['description']}" for c in candidates
    )

    return f"""You are a medical coding assistant. Given a clinical note and a list of candidate ICD-10-CM codes, select EVERY code that could reasonably apply to a condition described in the note.

            IMPORTANT RULES:
            - Only select codes from the candidate list below.
            - Lean toward INCLUSION: if a candidate plausibly matches a condition in the note, select it. Do not require perfect certainty — the official code wording may differ from the note's phrasing (e.g. "herniated disc" = "intervertebral disc displacement").
            - A case usually has SEVERAL applicable codes. It is better to include a plausible code than to miss a correct one.
            - Only reject a candidate if it clearly does not correspond to anything in the note.
            - Respond ONLY with the selected codes, one per line, no descriptions, no explanations.
            - Do not add any codes not in the candidate list.
            - Do not include any other text.

            Clinical Note:
            {medical_note}

            Candidate ICD-10-CM Codes:
            {candidate_list}

            Selected codes:"""


def extract_conditions(medical_note: str) -> list[str]:
    """
    First LLM pass: decompose the note into distinct diagnostic conditions.

    Each returned line becomes its own retrieval query. Keep it to actual
    diagnoses/findings — procedures and administrative items pull irrelevant
    codes into the pool and hurt precision without helping recall.
    """
    prompt = f"""You are a medical coding assistant. Read the clinical note and list each DISTINCT diagnosis, symptom, injury, or complication — the patient's medical CONDITIONS.

            RULES:
            - One condition per line.
            - Use precise, formal clinical terminology (prefer "intervertebral disc displacement" over "slipped disc").
            - List each distinct condition separately. Do NOT summarize or combine.
            - EXCLUDE procedures, surgeries, treatments, medications, monitoring, follow-up, and care-setting notes — list only the conditions being treated, not the actions taken.
            - No ICD codes, no numbering, no explanations — just the condition names.

            Clinical Note:
            {medical_note}

            Conditions:"""

    result = query_ollama(prompt)

    conditions = []
    for line in result["text"].strip().split("\n"):
        line = line.strip().strip("-•*").strip()
        if line:
            conditions.append(line)
    return conditions


def retrieve_for_conditions(
    conditions: list[str], collection, model, top_k: int = CANDIDATES_PER_CONDITION
) -> list[dict]:
    """
    Embed each condition (batched), retrieve top-K per condition in a single
    ChromaDB query, then merge: keep one entry per code with its lowest distance,
    sorted strongest-first for the final-selection LLM.
    """
    if not conditions:
        return []

    # One local encode call for all conditions; one batched Chroma query.
    embeddings = model.encode(conditions).tolist()
    results = collection.query(query_embeddings=embeddings, n_results=top_k)

    # Merge + dedup by code, keeping the minimum (closest) distance.
    best: dict[str, dict] = {}
    for qi in range(len(results["ids"])):          # one sub-list per condition
        for i in range(len(results["ids"][qi])):
            code = results["ids"][qi][i]
            dist = results["distances"][qi][i]
            if code not in best or dist < best[code]["distance"]:
                best[code] = {
                    "code": code,
                    "description": results["documents"][qi][i],
                    "distance": dist,
                }

    return sorted(best.values(), key=lambda c: c["distance"])


def query_ollama(prompt: str) -> dict:
    """
    Send prompt to Ollama (streaming) and return the response text + token usage.

    Streaming gives live progress and, together with num_predict, hard-bounds
    generation so a greedy repetition loop can't spin forever.
    """
    text_parts = []
    prompt_tokens = 0
    completion_tokens = 0

    with requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "think": False,  # disable qwen3 reasoning trace (the text token /nothink is ignored here)
            "options": {
                "temperature": 0.0,   # deterministic for reproducibility
                "num_predict": 100,   # cap output — codes are short; prevents runaway loops
            },
        },
        stream=True,
        timeout=300,  # don't hang forever if Ollama stalls
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("response", "")
            if piece:
                text_parts.append(piece)
                print(piece, end="", flush=True)  # live progress
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)
    print()  # newline after streamed output

    return {
        "text": "".join(text_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def parse_codes(llm_response: str) -> list[str]:
    """
    Extract ICD-10 codes from LLM response.
    Handles various formats: one per line, comma-separated, with/without dots.
    """
    codes = []
    # Split by newlines and commas
    for line in llm_response.strip().split("\n"):
        line = line.strip().strip("-•*").strip()
        if not line:
            continue
        # Take first token (in case LLM adds descriptions despite instructions)
        token = line.split()[0].strip(",").strip()
        # Remove dots for consistent comparison
        cleaned = token.replace(".", "").upper()
        if cleaned and len(cleaned) >= 3:
            codes.append(cleaned)
    return codes


def normalize_code(code: str) -> str:
    """Remove dots and uppercase for consistent comparison."""
    return code.replace(".", "").upper()


def compute_recall(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Recall = how many ground truth codes were found in predictions.
    """
    pred_set = set(normalize_code(c) for c in predicted)
    truth_set = set(normalize_code(c) for c in ground_truth)

    if not truth_set:
        return 1.0

    hits = truth_set.intersection(pred_set)
    return len(hits) / len(truth_set)


def compute_precision(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Precision = of the codes we predicted, how many were actually correct.
    """
    pred_set = set(normalize_code(c) for c in predicted)
    truth_set = set(normalize_code(c) for c in ground_truth)

    if not pred_set:
        return 0.0

    hits = truth_set.intersection(pred_set)
    return len(hits) / len(pred_set)


def run_pipeline():
    # Load cases
    cases = load_cases(CASES_PATH)
    if MAX_CASES is not None:  # TESTING ONLY — remove to run all cases
        cases = cases[:MAX_CASES]
    print(f"Loaded {len(cases)} cases")

    # Load embedding model and ChromaDB collection
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_collection()

    # Track metrics
    all_results = []
    total_recall = 0.0
    total_precision = 0.0
    total_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_gt_in_pool = 0  # ground truth codes that reached the candidate pool
    total_gt = 0          # ground truth codes overall (retrieval-recall denominator)

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i+1}")
        medical_note = case["medical_note"]
        ground_truth = case["icd10_cm"]["codes"]

        print(f"\n{'='*60}")
        print(f"Case {i+1}/{len(cases)}: {case_id}")
        print(f"Ground truth: {ground_truth}")

        start_time = time.time()

        # Step 1: Decompose the note into conditions, then retrieve per condition
        conditions = extract_conditions(medical_note)
        print(f"Extracted conditions ({len(conditions)}): {conditions}")
        candidates = retrieve_for_conditions(
            conditions, collection, model, top_k=CANDIDATES_PER_CONDITION
        )
        # Cap the pool sent to the LLM — candidates are sorted closest-first,
        # so this keeps the strongest and drops the long tail that confuses it.
        candidates = candidates[:MAX_CANDIDATES_FOR_LLM]
        print(f"Merged candidates: {len(candidates)}")

        # Step 2: Build prompt and query LLM for final code selection
        prompt = build_prompt(medical_note, candidates)

        # Diagnostic: check if ground truth codes appear in retrieved candidates
        candidate_codes = set(c["code"] for c in candidates)
        truth_codes = set(normalize_code(c) for c in ground_truth)
        found_in_candidates = truth_codes.intersection(candidate_codes)
        print(f"Ground truth in candidates: {len(found_in_candidates)}/{len(truth_codes)} — {found_in_candidates}")

        llm_result = query_ollama(prompt)
        elapsed = time.time() - start_time

        # Step 3: Parse predicted codes
        predicted = parse_codes(llm_result["text"])

        # Step 4: Compute recall + precision
        recall = compute_recall(predicted, ground_truth)
        precision = compute_precision(predicted, ground_truth)

        print(f"Predicted:    {predicted}")
        print(f"Recall:       {recall:.3f}")
        print(f"Precision:    {precision:.3f}")
        print(f"Time:         {elapsed:.2f}s")
        print(f"Tokens:       {llm_result['prompt_tokens']} prompt + {llm_result['completion_tokens']} completion")

        # Store result
        result = {
            "case_id": case_id,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "recall": recall,
            "precision": precision,
            "time_seconds": round(elapsed, 2),
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "llm_raw_response": llm_result["text"],
        }
        all_results.append(result)

        total_recall += recall
        total_precision += precision
        total_time += elapsed
        total_prompt_tokens += llm_result["prompt_tokens"]
        total_completion_tokens += llm_result["completion_tokens"]
        total_gt_in_pool += len(found_in_candidates)
        total_gt += len(truth_codes)

    # Summary
    n = len(cases)
    avg_recall = total_recall / n
    avg_precision = total_precision / n
    avg_time = total_time / n
    total_tokens = total_prompt_tokens + total_completion_tokens
    retrieval_recall = total_gt_in_pool / total_gt if total_gt else 0.0
    # Of the correct codes that reached the pool, what fraction did the LLM keep?
    llm_kept = (total_recall / n) / retrieval_recall if retrieval_recall else 0.0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Cases:                  {n}")
    print(f"Average recall:         {avg_recall:.3f}")
    print(f"Average precision:      {avg_precision:.3f}")
    print(f"Average time/case:      {avg_time:.2f}s")
    print(f"Total time:             {total_time:.1f}s")
    print(f"Total tokens:           {total_tokens} ({total_prompt_tokens} prompt + {total_completion_tokens} completion)")
    print(f"Avg tokens/case:        {total_tokens / n:.0f}")
    print(f"\n--- Two-stage recall ---")
    print(f"Retrieval recall:       {retrieval_recall:.3f}  ({total_gt_in_pool}/{total_gt} ground-truth codes reached the pool)")
    print(f"Final recall:           {avg_recall:.3f}  (codes the LLM actually selected)")
    print(f"LLM kept:               {llm_kept*100:.0f}% of recoverable codes")

    # Save detailed results
    output = {
        "summary": {
            "model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "candidates_per_condition": CANDIDATES_PER_CONDITION,
            "max_candidates_for_llm": MAX_CANDIDATES_FOR_LLM,
            "num_cases": n,
            "average_recall": round(avg_recall, 3),
            "average_precision": round(avg_precision, 3),
            "average_time_seconds": round(avg_time, 2),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "retrieval_recall": round(retrieval_recall, 3),
            "gt_in_pool": total_gt_in_pool,
            "gt_total": total_gt,
        },
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_pipeline()
