"""
Version A — retrieval-only baseline (NO final LLM selection).

Flow: note -> qwen3 extracts conditions -> embed each (S-PubMedBert) ->
vectorDB cosine search -> dedup/sort -> the retrieved codes ARE the prediction.

This is the baseline for the A/B comparison against Version B (fiftyfour_cases.py),
which adds a second LLM pass to select from a larger candidate pool.

Fair comparison requires BOTH recall and precision: a retrieval-only method can
trivially inflate recall by returning more codes, so recall alone is misleading.

Shared logic (extraction, retrieval, metrics) is imported from fiftyfour_cases
so the two versions cannot drift apart.

AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""

import json
import time

from sentence_transformers import SentenceTransformer

from retrieve_check import get_collection
from fiftyfour_cases import (
    load_cases,
    extract_conditions,
    retrieve_for_conditions,
    normalize_code,
    compute_recall,
)

# --- Config ---
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
CODES_PER_CONDITION = 1  # top-N codes returned per extracted condition (the cutoff)
MAX_CASES = 10  # TESTING ONLY — cap cases processed; set to None to run all 54
CASES_PATH = "data/icd10_cm_cases.json"
RESULTS_PATH = "data/results_retrieval_only.json"


def compute_precision(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Precision = of the codes we predicted, how many were actually correct.
    Guards against gaming recall by returning too many codes.
    """
    pred_set = set(normalize_code(c) for c in predicted)
    truth_set = set(normalize_code(c) for c in ground_truth)

    if not pred_set:
        return 0.0

    hits = truth_set.intersection(pred_set)
    return len(hits) / len(pred_set)


def run_pipeline():
    cases = load_cases(CASES_PATH)
    if MAX_CASES is not None:  # TESTING ONLY — remove to run all cases
        cases = cases[:MAX_CASES]
    print(f"Loaded {len(cases)} cases")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_collection()

    all_results = []
    total_recall = 0.0
    total_precision = 0.0
    total_time = 0.0

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i+1}")
        medical_note = case["medical_note"]
        ground_truth = case["icd10_cm"]["codes"]

        print(f"\n{'='*60}")
        print(f"Case {i+1}/{len(cases)}: {case_id}")
        print(f"Ground truth: {ground_truth}")

        start_time = time.time()

        # Step 1: Decompose the note into conditions (the only LLM call here)
        conditions = extract_conditions(medical_note)
        print(f"Extracted conditions ({len(conditions)}): {conditions}")

        # Step 2: Retrieve top-N codes per condition, dedup/sort — no second LLM.
        candidates = retrieve_for_conditions(
            conditions, collection, model, top_k=CODES_PER_CONDITION
        )
        elapsed = time.time() - start_time

        # The retrieved codes ARE the prediction.
        predicted = [c["code"] for c in candidates]

        recall = compute_recall(predicted, ground_truth)
        precision = compute_precision(predicted, ground_truth)

        print(f"Predicted ({len(predicted)}): {predicted}")
        print(f"Recall:    {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Time:      {elapsed:.2f}s")

        all_results.append({
            "case_id": case_id,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "recall": recall,
            "precision": precision,
            "time_seconds": round(elapsed, 2),
        })

        total_recall += recall
        total_precision += precision
        total_time += elapsed

    # Summary
    n = len(cases)
    avg_recall = total_recall / n
    avg_precision = total_precision / n
    avg_time = total_time / n

    print(f"\n{'='*60}")
    print(f"SUMMARY — Version A (retrieval only)")
    print(f"{'='*60}")
    print(f"Cases:              {n}")
    print(f"Average recall:     {avg_recall:.3f}")
    print(f"Average precision:  {avg_precision:.3f}")
    print(f"Average time/case:  {avg_time:.2f}s")
    print(f"Total time:         {total_time:.1f}s")

    output = {
        "summary": {
            "approach": "retrieval_only",
            "embedding_model": EMBEDDING_MODEL,
            "codes_per_condition": CODES_PER_CONDITION,
            "num_cases": n,
            "average_recall": round(avg_recall, 3),
            "average_precision": round(avg_precision, 3),
            "average_time_seconds": round(avg_time, 2),
        },
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_pipeline()
