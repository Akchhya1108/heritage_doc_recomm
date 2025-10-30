import os
import re
import json
import unicodedata
from bs4 import BeautifulSoup

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned data"
META_DIR = "data/metadata"

# ---------- TEXT CLEANING ----------
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[\u00A0]+", " ", text)
    return text.strip()

def clean_text(text):
    try:
        text = BeautifulSoup(text, "html.parser").get_text(separator="\n")
    except Exception:
        pass
    text = normalize_text(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[_*`~=]{2,}", "", text)
    text = re.sub(r"([-=_+]{3,})", "", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln != ""])
    return text.strip()

# ---------- HELPERS ----------
def detect_source(path):
    n = path.lower()
    if "wiki" in n or "wikipedia" in n:
        return "Wikipedia"
    if "unesco" in n:
        return "UNESCO"
    if "asi" in n or "archaeolog" in n or "indi" in n:
        return "Indian Heritage"
    if "archive" in n:
        return "Archive.org"
    return "Unknown"

# ---------- PROCESSOR ----------
def process_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    per_doc_meta = []

    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ Skipped {fname} (JSON error: {e})")
                continue

            title = data.get("title", "Untitled")
            content = data.get("content", "")
            summary = data.get("summary", "")
            combined = f"{title}\n\n{summary}\n\n{content}"
            cleaned = clean_text(combined)

            rel_folder = os.path.relpath(root, RAW_DIR)
            clean_subdir = os.path.join(CLEAN_DIR, rel_folder)
            os.makedirs(clean_subdir, exist_ok=True)

            cleaned_fname = fname.replace(".json", ".txt")
            cleaned_path = os.path.join(clean_subdir, cleaned_fname)
            with open(cleaned_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            meta = {
                "file_name": fname,
                "title": title,
                "source": data.get("source", detect_source(fname)),
                "url": data.get("url", ""),
                "categories": data.get("categories", []),
                "document_type": data.get("metadata", {}).get("document_type", ""),
                "topic": data.get("metadata", {}).get("topic", ""),
                "word_count": len(cleaned.split()),
                "char_count": len(cleaned),
                "raw_path": fpath,
                "cleaned_path": cleaned_path
            }
            per_doc_meta.append(meta)

    combined_path = os.path.join(META_DIR, "metadata.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(per_doc_meta, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned {len(per_doc_meta)} files")
    print(f"🧹 Cleaned texts -> {CLEAN_DIR}")
    print(f"🗂 Metadata -> {combined_path}")

if __name__ == "__main__":
    process_all()
