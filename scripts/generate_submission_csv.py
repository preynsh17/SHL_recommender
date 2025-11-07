import os, time, pandas as pd, requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

API_URL = "https://shl-recommender-1-92qj.onrender.com/recommend"
XLSX_PATH = "Gen_AI Dataset.xlsx"
OUT_CSV = "submission.csv"
TOP_K = 1

def session():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.8, status_forcelist=[502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

def read_queries(path):
    df = pd.read_excel(path)
    candidates = {"query", "queries", "jd", "job description", "prompt", "text"}
    col = next((c for c in df.columns if str(c).strip().lower() in candidates), df.columns[0])
    return [str(x).strip() for x in df[col].dropna().tolist()]

def main():
    s = session()
    queries = read_queries(XLSX_PATH)
    rows = []

    print(f"\n🔍 Found {len(queries)} queries. Starting requests to {API_URL}\n")

    for q in tqdm(queries, desc="Processing Queries", unit="query"):
        try:
            r = s.post(API_URL, json={"query": q, "top_k": TOP_K}, timeout=90)
            r.raise_for_status()
            results = r.json().get("results", [])
            url = results[0]["url"] if results else ""
            rows.append({"Query": q, "Assessment_url": url})
        except Exception as e:
            tqdm.write(f"⚠️ Error for query: {q[:50]}... ({e})")
            rows.append({"Query": q, "Assessment_url": ""})
        time.sleep(0.2)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n✅ Done! Saved results → {OUT_CSV}\n")

if __name__ == "__main__":
    main()
