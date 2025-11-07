# index/search_engine.py
import os, numpy as np, pickle
from pathlib import Path

BM25_PATH = Path("index/bm25_index.pkl")
META_PATH = Path("index/meta.pkl")
SEMANTIC = os.environ.get("SEMANTIC", "1") == "1"

with open(BM25_PATH, "rb") as f: bm25 = pickle.load(f)
with open(META_PATH, "rb") as f: meta = pickle.load(f)

if SEMANTIC:
    import faiss
    FAISS_PATH = Path("index/faiss_index.bin")
    index = faiss.read_index(str(FAISS_PATH))
    _model = None
    def get_model():
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model

def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or np.ptp(x) == 0: return np.zeros_like(x)
    return (x - x.min()) / (np.ptp(x) + 1e-9)

def _bm25(query):
    toks = query.lower().split()
    scores = _normalize(bm25.get_scores(toks))
    order = np.argsort(scores)[::-1]
    return order, scores

def hybrid_search(query, top_k=10, w_semantic=0.7):
    order, bm = _bm25(query)
    if not SEMANTIC:
        top = order[:top_k]
        return [{"name": meta["titles"][i], "url": meta["urls"][i],
                 "bm25": float(bm[i]), "semantic": 0.0,
                 "combined_score": float(bm[i])} for i in top]

    q = get_model().encode([query], normalize_embeddings=True)
    D, I = index.search(q, len(meta["titles"]))
    sem = np.zeros_like(bm)
    sem[I[0]] = _normalize(D[0])
    combined = w_semantic * sem + (1 - w_semantic) * bm
    top = np.argsort(combined)[::-1][:top_k]
    return [{"name": meta["titles"][i], "url": meta["urls"][i],
             "bm25": float(bm[i]), "semantic": float(sem[i]),
             "combined_score": float(combined[i])} for i in top]
