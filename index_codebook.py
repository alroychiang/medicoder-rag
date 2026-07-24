# AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""
Index pre-computed embeddings into ChromaDB.
Reads the cleaned icd10_billable_codes.jsonl and corresponding embeddings.npy,
then inserts into a persistent ChromaDB collection.

AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""

import json
import numpy as np
import chromadb

COLLECTION_NAME = "icd10_codes"
BATCH_SIZE = 500


def load_codes(jsonl_path: str) -> list[dict]:
    codes = []
    with open(jsonl_path, "r") as f:
        for line in f:
            codes.append(json.loads(line))
    return codes


def index(
    jsonl_path: str = "data/icd10_billable_codes.jsonl",
    embeddings_path: str = "data/embeddings.npy",
    db_path: str = "chroma_db",
) -> None:
    # Load codes and embeddings
    codes = load_codes(jsonl_path)
    embeddings = np.load(embeddings_path)

    assert len(codes) == len(embeddings), (
        f"Mismatch: {len(codes)} codes but {len(embeddings)} embeddings"
    )
    print(f"Loaded {len(codes)} codes, embeddings shape {embeddings.shape}")

    # Init ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # Recreate collection from scratch (idempotent)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches
    code_ids = [c["code"] for c in codes]
    descriptions = [c["description"] for c in codes]

    for i in range(0, len(codes), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(codes))
        collection.add(
            ids=code_ids[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=descriptions[i:batch_end],
            metadatas=[{"code": c["code"]} for c in codes[i:batch_end]],
        )

    print(f"Indexed {collection.count()} codes in ChromaDB at '{db_path}'")

    # Peek at one record to verify
    peek = collection.peek(limit=1)
    print(f"\nSample record:")
    print(f"  id:       {peek['ids'][0]}")
    print(f"  document: {peek['documents'][0][:80]}...")
    print(f"  metadata: {peek['metadatas'][0]}")


if __name__ == "__main__":
    index()
