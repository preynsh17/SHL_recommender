# index/test_index.py
import json, pickle, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# ---------- paths ----------
BM25_PATH = "index/bm25_index.pkl"
FAISS_PATH = "index/faiss_index.bin"
EMB_PATH = "index/embeddings.npy"
META_PATH = "index/meta.pkl"
CORPUS_PATH = "index/corpus.jsonl"

# ---------- load ----------
print("📂 Loading indexes...")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)
index = faiss.read_index(FAISS_PATH)
embeddings = np.load(EMB_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- helpers ----------
def hybrid_search(query, top_k=10, w_semantic=0.7):
    """Combine FAISS semantic similarity and BM25 lexical scores"""
    # BM25
    query_tok = query.lower().split()
    bm25_scores = bm25.get_scores(query_tok)
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-9)

    # FAISS
    q_emb = model.encode([query], normalize_embeddings=True)
    _, idxs = index.search(q_emb, len(meta["titles"]))
    faiss_scores = np.zeros_like(bm25_scores)
    faiss_scores[idxs[0]] = np.linspace(1, 0, len(idxs[0]))

    # Combine
    combined = w_semantic * faiss_scores + (1 - w_semantic) * bm25_scores
    top_idx = np.argsort(combined)[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "name": meta["titles"][i],
            "url": meta["urls"][i],
            "bm25": float(bm25_scores[i]),
            "semantic": float(faiss_scores[i]),
            "combined": float(combined[i])
        })
    return results

# ---------- main ----------
if __name__ == "__main__":
    print("✅ Indexes loaded successfully.")
    while True:
        q = input("\n🔎 Enter your query (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        res = hybrid_search(q, top_k=10)
        print("\nTop results:")
        for r in res:
            print(f"- {r['name']} ({r['url']}) | Score: {r['combined']:.4f}")
