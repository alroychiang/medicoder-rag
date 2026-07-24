# AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""
Retrieve candidate ICD-10 codes from ChromaDB given a medical note.
Uses the same embedding model as embed_codebook.py to encode the query,
then performs cosine similarity search against the indexed code descriptions.

AI-assisted: scaffold generated with Claude, reviewed and understood by author.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
COLLECTION_NAME = "icd10_codes"


def get_collection(db_path: str = "chroma_db"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, collection, model, top_k: int = 20) -> list[dict]:
    """
    Embed a medical note and retrieve top-K candidate codes from ChromaDB.

    Returns list of dicts: [{"code": "T18128A", "description": "...", "distance": 0.23}, ...]
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    # retrieved chunks
    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "code": results["ids"][0][i],
            "description": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })

    return candidates


# Quick sanity check — run this file directly to test retrieval
if __name__ == "__main__":
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_collection()

    test_query = "Patient presented with food stuck in esophagus after eating steak"
    print(f"\nQuery: {test_query}")
    print(f"Top 10 candidates:\n")

    candidates = retrieve(test_query, collection, model, top_k=10)
    for i, c in enumerate(candidates):
        print(f"  {i+1}. {c['code']:10s} (dist: {c['distance']:.4f})  {c['description']}")
