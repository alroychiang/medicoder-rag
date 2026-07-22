"""
Parse the ICD-10-CM codebook text file into a clean JSONL of billable codes only.
Filters out header/category codes (flag 0), keeps only assignable codes (flag 1).

AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""

import json
import sys


def parse_codebook(input_path: str, output_path: str) -> None:
    total = 0
    billable = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            total += 1
            # Fixed-width format:
            # [0:5]   order number
            # [6:13]  code (no dots)
            # [14:15] billable flag: 1 = assignable, 0 = category header
            # [16:77] short description
            # [77:]   long description
            code = line[6:13].strip()
            flag = line[14:15].strip()
            long_desc = line[77:].strip()

            if flag != "1":
                continue

            billable += 1
            entry = {"code": code, "description": long_desc}
            fout.write(json.dumps(entry) + "\n")

    print(f"Total lines: {total}")
    print(f"Billable codes written: {billable}")
    print(f"Headers skipped: {total - billable}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/icd10cm_order_2026.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/icd10_billable_codes.jsonl"
    parse_codebook(input_path, output_path)