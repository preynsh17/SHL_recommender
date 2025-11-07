# crawler/scrape_catalog.py
import re, time, json, pathlib
from urllib.parse import urlencode, urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
from playwright.sync_api import sync_playwright

BASE = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE = 12
TOTAL_PAGES = 32
OUT_JSONL = pathlib.Path("data/raw/catalog.jsonl")

def ensure_dirs():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

def listing_url(start):
    # type=1 => Individual Test Solutions
    return f"{BASE}?{urlencode({'start': start, 'type': 1})}"

def accept_cookies(page):
    for sel in [
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        ".cookie-accept", ".optanon-allow-all"
    ]:
        try:
            page.locator(sel).first.click(timeout=800)
            break
        except Exception:
            pass

def wait_for_catalog(page):
    # Wait for the table/rows to render
    try:
        page.wait_for_selector("td.custom__table-heading__title a, a.pagination__arrow", timeout=10000)
    except Exception:
        time.sleep(1)

def collect_listing_links(html):
    # Most reliable per your inspect: links live inside td.custom__table-heading__title
    soup = BeautifulSoup(html, "html.parser")
    hrefs = []
    for a in soup.select('td.custom__table-heading__title a[href]'):
        h = a.get("href", "")
        if not h or h.startswith("#"):
            continue
        # Only keep the “view” detail pages from catalog
        if "/products/product-catalog/view/" in h:
            hrefs.append(urljoin(BASE, h))
    # Dedup while keeping order
    seen, out = set(), []
    for h in hrefs:
        if h not in seen:
            seen.add(h); out.append(h)
    return out

def extract_text(soup):
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

def guess_test_type(text):
    low = text.lower()
    if "personality & behavior" in low or "personality and behavior" in low or "behavioral" in low:
        return "Personality & Behavior"
    if "knowledge & skills" in low or "knowledge and skills" in low or "knowledge" in low or "skills" in low:
        return "Knowledge & Skills"
    m = re.search(r"\b(K|P)\b", text)
    return {"K": "Knowledge & Skills", "P": "Personality & Behavior"}.get(m.group(1)) if m else None

def is_prepackaged(text):
    low = text.lower()
    return "pre-packaged job solution" in low or "prepackaged job solution" in low

def crawl_catalog():
    ensure_dirs()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        context = browser.new_context()
        page = context.new_page()

        print("📄 Collecting product links from listing pages...")
        all_links = []
        for i in tqdm(range(TOTAL_PAGES), desc="Listing pages"):
            start = i * PAGE_SIZE
            url = listing_url(start)
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            accept_cookies(page)
            wait_for_catalog(page)
            all_links.extend(collect_listing_links(page.content()))

        # Deduplicate across pages
        all_links = list(dict.fromkeys(all_links))
        print(f"✅ Found {len(all_links)} unique catalog detail links")

        print("🔍 Scraping detail pages...")
        rows = []
        for u in tqdm(all_links, desc="Detail pages"):
            try:
                page.goto(u, wait_until="domcontentloaded", timeout=60000)
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                # Title: prefer h1; fall back to first heading
                title = None
                for tag in ["h1", "h2", "h3"]:
                    t = soup.find(tag)
                    if t and t.get_text(strip=True):
                        title = t.get_text(strip=True)
                        break
                if not title:
                    # final fallback: use the anchor text the list used (last URL segment)
                    title = u.rstrip("/").split("/")[-1].replace("-", " ").title()

                text = extract_text(soup)
                if is_prepackaged(text):
                    continue
                tt = guess_test_type(text)

                rows.append({
                    "name": title,
                    "url": u,                 # we can keep this “view” URL as the official link
                    "test_type": tt,
                    "raw_text": text
                })
            except Exception:
                continue

        browser.close()

    # Write JSONL corpus
    seen = set()
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(seen)} assessments to {OUT_JSONL}")

if __name__ == "__main__":
    crawl_catalog()
