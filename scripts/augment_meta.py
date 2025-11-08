# scripts/augment_meta.py
import json, re, pickle
from pathlib import Path

SRC = Path("data/raw/catalog.jsonl")
OUT = Path("index/meta.pkl")

def norm_space(s): return re.sub(r"\s+", " ", s or "").strip()

def parse_duration(text):
    t = text.lower()
    m = re.search(r"(\d+)\s*(minutes|min)\b", t)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s*(hours|hour|hrs|hr)\b", t)
    if m: return int(m.group(1)) * 60
    return None

LEVEL_KEYS = ["Director","Entry-Level","Executive","General Population","Graduate","Manager","Mid-Professional","Front Line Manager","Supervisor"]
TYPE_MAP = {
    "Ability & Aptitude":"A","Biodata & Situational Judgement":"B","Competencies":"C","Development & 360":"D",
    "Assessment Exercises":"E","Knowledge & Skills":"K","Personality & Behavior":"P","Simulations":"S"
}

def parse_levels(text):
    found = []
    for k in LEVEL_KEYS:
        if re.search(rf"\b{re.escape(k)}\b", text, flags=re.I):
            found.append(k)
    return sorted(set(found))

def parse_test_type(text):
    # Prefer full names; fall back to letter code if only like "Test Type: A ..."
    for name, code in TYPE_MAP.items():
        if re.search(rf"\b{name}\b", text, flags=re.I): return {"name":name, "code":code}
    m = re.search(r"Test Type:\s*([A-Z])\b", text)
    if m:
        code = m.group(1)
        name = next((n for n,c in TYPE_MAP.items() if c==code), None)
        return {"name": name or "", "code": code}
    return {"name":"", "code":""}

def load_jsonl(p):
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        yield json.loads(line)

def main():
    titles, urls, raw_texts, levels, test_types, durations = [], [], [], [], [], []
    for item in load_jsonl(SRC):
        name = norm_space(item.get("name",""))
        url = norm_space(item.get("url",""))
        raw = norm_space(item.get("raw_text",""))
        tt = parse_test_type(raw)
        lv = parse_levels(raw)
        dur = parse_duration(raw)
        titles.append(name)
        urls.append(url)
        raw_texts.append(raw)
        levels.append(lv)
        test_types.append(tt)
        durations.append(dur)

    meta = {
        "titles": titles,
        "urls": urls,
        "raw_texts": raw_texts,
        "job_levels": levels,
        "test_types": test_types,     # dicts: {"name","code"}
        "duration_min": durations     # int or None
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f: pickle.dump(meta, f)
    print(f"saved meta with {len(titles)} items -> {OUT}")

if __name__ == "__main__":
    main()
