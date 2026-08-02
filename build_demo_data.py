# AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""
Merge evaluation cases with pipeline results into a single file for the demo viewer.

The two source files line up by POSITION, not by key: the cases file has no
case_id field, so fiftyfour_cases.py assigns "case_{i+1}" from the loop index.
We zip them back together and assert the ground truth matches on both sides —
that assertion is what makes the positional join safe rather than assumed.

Outputs:
  demo/demo_data.json  — notes + predictions + metrics, one entry per case
  demo/index.html      — self-contained viewer with that data inlined

The data is inlined rather than fetched so the page works from file:// and on any
static host, with no CORS or asset-path setup.
"""

import json
from pathlib import Path

CASES_PATH = Path("data/icd10_cm_cases.json")
RESULTS_PATH = Path("data/results_full.json")
OUTPUT_PATH = Path("demo/demo_data.json")
TEMPLATE_PATH = Path("demo/template.html")
PAGE_PATH = Path("demo/index.html")


def normalize_code(code: str) -> str:
    """Remove dots and uppercase — matches the pipeline's comparison logic."""
    return code.replace(".", "").upper()


def build() -> dict:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    results_file = json.loads(RESULTS_PATH.read_text())
    results = results_file["results"]

    if len(cases) != len(results):
        raise SystemExit(
            f"Case/result count mismatch: {len(cases)} cases vs {len(results)} results. "
            "Re-run fiftyfour_cases.py over the full dataset before building."
        )

    merged = []
    for i, (case, result) in enumerate(zip(cases, results)):
        # The join is positional — verify it actually holds before trusting it.
        case_truth = {normalize_code(c) for c in case["icd10_cm"]["codes"]}
        result_truth = {normalize_code(c) for c in result["ground_truth"]}
        if case_truth != result_truth:
            raise SystemExit(
                f"Ground truth mismatch at index {i} ({result['case_id']}).\n"
                f"  cases file:   {sorted(case_truth)}\n"
                f"  results file: {sorted(result_truth)}\n"
                "The two files are out of order — regenerate results_full.json."
            )

        predicted = [normalize_code(c) for c in result["predicted"]]
        truth = [normalize_code(c) for c in result["ground_truth"]]

        merged.append({
            "case_id": result["case_id"],
            "medical_note": case["medical_note"],
            "ground_truth": truth,
            "predicted": predicted,
            "hits": sorted(set(truth) & set(predicted)),      # correctly predicted
            "missed": sorted(set(truth) - set(predicted)),    # in truth, not predicted
            "spurious": sorted(set(predicted) - set(truth)),  # predicted, not in truth
            "recall": result["recall"],
            "precision": result["precision"],
            "time_seconds": result["time_seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        })

    return {"summary": results_file["summary"], "cases": merged}


def render_page(data: dict) -> str:
    """Inline the dataset into the template as a JS literal."""
    template = TEMPLATE_PATH.read_text()
    if "/*__DATA__*/ null" not in template:
        raise SystemExit(f"{TEMPLATE_PATH} is missing the '/*__DATA__*/ null' placeholder.")

    # Escape '<' so a '</script>' inside any note can't close the tag early.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("/*__DATA__*/ null", payload)


if __name__ == "__main__":
    data = build()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))
    PAGE_PATH.write_text(render_page(data))

    n = len(data["cases"])
    print(f"Merged {n} cases -> {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"Ground truth verified on all {n} cases (positional join holds)")
    print(f"Rendered viewer -> {PAGE_PATH} ({PAGE_PATH.stat().st_size / 1024:.0f} KB, self-contained)")
