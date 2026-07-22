"""
Embed billable ICD-10-CM code descriptions into vectors.
Reads the cleaned .jsonl from parse_codebook.py, embeds using
sentence-transformers, and saves vectors to .npy.

AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_codes(jsonl_path: str) -> list[dict]:
    codes = []
    with open(jsonl_path, "r") as f:
        for line in f:
            codes.append(json.loads(line))
    return codes


def embed(jsonl_path: str, output_path: str = "data/embeddings.npy") -> None:
    codes = load_codes(jsonl_path)
    descriptions = [c["description"] for c in codes]
    print(f"Loaded {len(descriptions)} descriptions")

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding...")
    start = time.time()
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=256)
    elapsed = time.time() - start

    np.save(output_path, embeddings)
    print(f"Embedding took {elapsed:.1f}s")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    embed("data/icd10_billable_codes.jsonl")
