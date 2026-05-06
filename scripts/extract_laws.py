#!/usr/bin/env python3
"""
scripts/extract_laws.py
Syrian Personal Status Law — TXT → JSON.

Reads the raw text file from data/raw/ and produces a structured JSON
file in data/processed/ with one object per article.

Usage
-----
    python scripts/extract_laws.py
"""

import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED

INPUT_FILE  = DATA_RAW       / "personal_status_law.txt"
OUTPUT_FILE = DATA_PROCESSED / "laws.json"

LAW_NAME = "قانون الأحوال الشخصية السوري"
LAW_YEAR = 1953
LAW_TYPE = "substantive"

# Eastern Arabic numerals → ASCII digits
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def _normalize_num(text: str) -> str:
    return text.translate(_ARABIC_INDIC)

RE_KITAB   = re.compile(r"^الكتاب\s+.+", re.U)
RE_BAB     = re.compile(r"^\s*الباب\s+.+", re.U)
RE_FASL    = re.compile(r"^\s*الفصل\s+.+", re.U)
RE_QISM    = re.compile(r"^\s*القسم\s+.+", re.U)
RE_ARTICLE = re.compile(r"^المادة\s+([\d٠-٩]+(?:\s*-\s*مكرر)?)\s*$", re.U)

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "زواج":        ["زواج", "عقد الزواج", "خطبة", "مهر", "ولاية", "كفاءة"],
    "طلاق":        ["طلاق", "فسخ", "مخالعة", "تفريق", "عدة", "رجعة"],
    "نفقة":        ["نفقة", "مؤنة", "إعالة", "نفقات"],
    "حضانة":       ["حضانة", "رؤية", "مشاهدة"],
    "نسب":         ["نسب", "بنوة", "ولادة", "إقرار بالنسب"],
    "وصاية":       ["وصي", "وصاية", "قيّم", "قيمومة"],
    "ميراث":       ["ميراث", "إرث", "تركة", "وصية", "موروث"],
    "أهلية":       ["أهلية", "قاصر", "بلوغ", "رشد", "محجور"],
    "أحوال_مدنية": ["نفوس", "سجل", "وثيقة", "شهادة ولادة", "جنسية"],
}


def _extract_topics(text: str) -> list[str]:
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in text for kw in kws)]


def extract_laws(input_path: Path) -> list[dict]:
    """Parse the raw law text file and return a list of article dicts."""
    with open(input_path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    articles: list[dict] = []
    current_kitab = current_bab = current_fasl = current_qism = ""
    current_article_num:   str | None  = None
    current_article_lines: list[str]   = []
    in_articles = False

    def _flush() -> None:
        if current_article_num is None:
            return
        text = "\n".join(current_article_lines).strip()
        if not text:
            return
        num_norm  = _normalize_num(current_article_num).strip()
        is_mokrar = "مكرر" in num_norm
        num_clean = re.sub(r"\s*-\s*مكرر", "", num_norm).strip()
        try:
            num_int = int(num_clean)
        except ValueError:
            num_int = 0

        article_id = f"law_احوال_{num_clean}" + ("_مكرر" if is_mokrar else "")
        articles.append({
            "id":                 article_id,
            "article_number":     num_int,
            "article_number_str": f"المادة {num_clean}" + (" - مكرر" if is_mokrar else ""),
            "text": text,
            "metadata": {
                "law_name": LAW_NAME,
                "law_year": LAW_YEAR,
                "type":     LAW_TYPE,
                "kitab":    current_kitab.strip(),
                "bab":      current_bab.strip(),
                "fasl":     current_fasl.strip(),
                "qism":     current_qism.strip(),
                "topics":   _extract_topics(text),
            },
        })

    for line in lines:
        stripped = line.strip()
        if RE_KITAB.match(stripped):
            current_kitab = stripped
            current_bab = current_fasl = current_qism = ""
            continue
        if RE_BAB.match(stripped):
            current_bab = stripped
            current_fasl = current_qism = ""
            continue
        if RE_FASL.match(stripped):
            current_fasl = stripped
            current_qism = ""
            continue
        if RE_QISM.match(stripped):
            current_qism = stripped
            continue
        m = RE_ARTICLE.match(stripped)
        if m:
            _flush()
            current_article_num   = m.group(1)
            current_article_lines = []
            in_articles = True
            continue
        if in_articles and current_article_num is not None:
            current_article_lines.append(line)

    _flush()
    return articles


def main() -> None:
    print(f"[1] Reading {INPUT_FILE} ...")
    articles = extract_laws(INPUT_FILE)
    print(f"[2] Extracted {len(articles)} articles")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[3] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
