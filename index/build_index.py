# index/build_index.py
import json, re, pickle, numpy as np, pandas as pd
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

RAW_PATH = Path("data/raw/catalog.jsonl")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers --------------------------------------------------------

def clean_text(text):
    # Trim boilerplate: remove excess whitespace and obvious footer noise
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"©\s*\d{4}\s*SHL.*", "", t)
    t = t.strip()
    return t

def load_catalog():
    rows = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("name") and obj.get("raw_text"):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)

# --- Main -----------------------------------------------------------

print("📂 Loading catalog...")
df = load_catalog()
print(f"Loaded {len(df)} records")

# Clean & prepare text
df["text"] = df["raw_text"].apply(clean_text)
df["text_short"] = df["text"].str.slice(0, 3000)   # keep first 3000 chars

corpus = df["text_short"].tolist()
titles = df["name"].tolist()
urls = df["url"].tolist()

# Save corpus for reference
with (INDEX_DIR / "corpus.jsonl").open("w", encoding="utf-8") as f:
    for t, u, x in zip(titles, urls, corpus):
        f.write(json.dumps({"name": t, "url": u, "text": x}) + "\n")

# --- BM25 -----------------------------------------------------------
print("🔍 Building BM25 index...")
tokenized = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)
with open(INDEX_DIR / "bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

# --- Embeddings + FAISS --------------------------------------------
print("⚙️ Generating embeddings (MiniLM)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(corpus, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

print("💾 Saving FAISS index...")
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))
np.save(INDEX_DIR / "embeddings.npy", embeddings)

# metadata
meta = {"titles": titles, "urls": urls}
with open(INDEX_DIR / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("✅ Index built successfully!")
print(f"→ {len(corpus)} items embedded and indexed")
