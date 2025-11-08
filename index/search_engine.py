import os, re, numpy as np, pickle
from pathlib import Path

BM25_PATH = Path("index/bm25_index.pkl")
FAISS_PATH = Path("index/faiss_index.bin")
META_PATH = Path("index/meta.pkl")

SEMANTIC = os.environ.get("SEMANTIC", "1") == "1"

# --- Load indexes ---
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

if SEMANTIC:
    import faiss
    index = faiss.read_index(str(FAISS_PATH))
    _model = None

    def get_model():
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model


# --- Utility functions ---
def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (np.ptp(x) + 1e-9)


def _bm25(query):
    toks = query.lower().split()
    scores = _normalize(bm25.get_scores(toks))
    order = np.argsort(scores)[::-1]
    return order, scores


# --- Keyword dictionaries ---
ROLE_HINTS = {
    "graduate": ["graduate", "fresher", "entry", "campus"],
    "manager": ["manager", "lead", "supervisor", "team lead", "people manager"],
    "executive": ["executive", "director", "cxo", "coo", "cto", "ceo", "vp", "senior leader"],
}
DOMAIN_HINTS = {
    "sales": ["sales", "seller", "selling", "pipeline", "bd", "business development", "account executive"],
    "engineering": ["engineer", "developer", "programmer", "software", "java", "python", "c++", "coding"],
    "customer": ["support", "contact center", "call center", "bpo"],
}
TYPE_HINTS = {
    "technical": ["java", "python", "c++", "coding", "developer", "engineer", "programming"],
    "behavioral": ["culture", "fit", "values", "personality", "communication", "collaborate", "interpersonal"],
}
CULTURE_HINTS = ["culture", "cultural", "values", "fit", "behavioral", "personality", "leadership style"]


# --- Parse constraints from natural language query ---
def parse_constraints(q):
    ql = q.lower()
    dur = None
    m = re.search(r"(\d+)\s*(minutes|min)\b", ql)
    if m:
        dur = int(m.group(1))
    m = re.search(r"(\d+)\s*(hours|hour|hrs|hr)\b", ql)
    if m:
        dur = int(m.group(1)) * 60

    level = None
    if any(w in ql for w in ROLE_HINTS["graduate"]):
        level = "Graduate"
    elif any(w in ql for w in ROLE_HINTS["manager"]):
        level = "Manager"
    elif any(w in ql for w in ROLE_HINTS["executive"]):
        level = "Executive"

    domain = None
    for k, words in DOMAIN_HINTS.items():
        if any(w in ql for w in words):
            domain = k
            break

    desired_type = None
    if any(w in ql for w in TYPE_HINTS["technical"]):
        desired_type = "technical"
    elif any(w in ql for w in TYPE_HINTS["behavioral"]):
        desired_type = "behavioral"

    culture = any(w in ql for w in CULTURE_HINTS)

    return {
        "duration": dur,
        "level": level,
        "domain": domain,
        "desired_type": desired_type,
        "culture": culture,
    }


# --- Metadata-aware reranker ---
def metadata_boost(
    idx_list, base_scores, constraints, w_level=0.18, w_duration=0.18, w_type=0.22
):
    boosts = np.zeros_like(base_scores)
    dur = constraints.get("duration")
    level = constraints.get("level")
    desired_type = constraints.get("desired_type")

    for i in idx_list:
        b = 0.0

        # --- job level boost ---
        if level and level in (meta["job_levels"][i] or []):
            b += w_level

        # --- duration closeness ---
        if dur and (meta["duration_min"][i] is not None):
            diff = abs(meta["duration_min"][i] - dur)
            if diff <= 10:
                b += w_duration
            elif diff <= 20:
                b += w_duration * 0.5

        # --- test type alignment ---
        if desired_type:
            tt = meta["test_types"][i] or {}
            name = (tt.get("name") or "").lower()
            code = (tt.get("code") or "").upper()
            is_tech = ("technical" in name) or (code in {"K"})
            is_behav = (
                ("personality" in name)
                or ("behavior" in name)
                or ("competenc" in name)
                or ("360" in name)
                or (code in {"P", "C", "D"})
            )

            if desired_type == "technical" and is_tech:
                b += w_type
            if desired_type == "behavioral" and is_behav:
                b += w_type
            # mismatch penalty
            if desired_type == "behavioral" and is_tech:
                b -= w_type * 0.4
            if desired_type == "technical" and is_behav:
                b -= w_type * 0.2

        # --- culture fit preference ---
        if constraints.get("culture"):
            tt = meta["test_types"][i] or {}
            name = (tt.get("name") or "").lower()
            code = (tt.get("code") or "").upper()
            if ("personality" in name or "behavior" in name or "360" in name) or code in {
                "P",
                "C",
                "D",
            }:
                b += 0.15

        # --- domain/level hints (sales, graduate) ---
        title_l = (meta["titles"][i] or "").lower()
        if constraints.get("level") == "Graduate" and "graduate" in title_l:
            b += 0.1
        if constraints.get("domain") == "sales" and "sales" in title_l:
            b += 0.1

        boosts[i] = b

    return boosts


# --- Hybrid search combining semantic + BM25 + metadata rerank ---
def hybrid_search(query, top_k=10, w_semantic=0.7):
    order, bm = _bm25(query)
    bm_norm = bm
    cons = parse_constraints(query)

    # --- BM25 only mode (no FAISS) ---
    if not SEMANTIC:
        cand = order[: max(top_k * 8, 50)]
        boost = metadata_boost(cand, bm_norm, cons)
        combined = bm_norm + boost
        top = np.argsort(combined)[::-1][:top_k]
        return [
            {
                "name": meta["titles"][i],
                "url": meta["urls"][i],
                "bm25": float(bm_norm[i]),
                "semantic": 0.0,
                "combined_score": float(combined[i]),
            }
            for i in top
        ]

    # --- Semantic + BM25 hybrid ---
    q = get_model().encode([query], normalize_embeddings=True)
    D, I = index.search(q, len(meta["titles"]))
    sem = np.zeros_like(bm_norm)
    sem[I[0]] = _normalize(D[0])

    combined = w_semantic * sem + (1 - w_semantic) * bm_norm
    cand = np.argsort(combined)[::-1][: max(top_k * 8, 100)]
    boost = metadata_boost(cand, combined, cons)
    combined2 = combined + boost
    top = np.argsort(combined2)[::-1][:top_k]

    return [
        {
            "name": meta["titles"][i],
            "url": meta["urls"][i],
            "bm25": float(bm_norm[i]),
            "semantic": float(sem[i]),
            "combined_score": float(combined2[i]),
        }
        for i in top
    ]
