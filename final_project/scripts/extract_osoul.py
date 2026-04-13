#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/extract_osoul.py
قانون أصول المحاكمات المدنية السوري — TXT → JSON

يقرأ الملف النصي من data/raw/ ويُنتج JSON في data/processed/

الاستخدام:
    python scripts/extract_osoul.py
"""

import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED

INPUT_FILE  = DATA_RAW       / "civil_procedure_law.txt"
OUTPUT_FILE = DATA_PROCESSED / "osoul.json"

LAW_NAME = "قانون أصول المحاكمات المدنية السوري"
LAW_YEAR = 2016
LAW_TYPE = "procedural"

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_num(text: str) -> str:
    return text.translate(ARABIC_INDIC)

RE_ARTICLE = re.compile(r'^\s{0,4}المادة\s*([\d٠-٩]+(?:\s*مكرر)?)\s*$', re.U)
RE_BAB     = re.compile(r'^\s*باب\s*.+|^باب\s*.+', re.U)
RE_FASL    = re.compile(r'^\s*الفصل\s+.+|^فصل\s*.+', re.U)
RE_QISM    = re.compile(r'^\s*القسم\s+.+', re.U)
RE_FARA    = re.compile(r'^\s*الفرع\s+.+', re.U)

TOPIC_KEYWORDS = {
    "اختصاص":       ["اختصاص", "محكمة مختصة", "صلاحية"],
    "دعوى":         ["دعوى", "لائحة", "عريضة", "مدعي", "مدعى عليه"],
    "تبليغ":        ["تبليغ", "إخبار", "إعلام", "محضر تبليغ"],
    "جلسة":         ["جلسة", "موعد", "حضور", "غياب"],
    "إثبات":        ["إثبات", "بينة", "شاهد", "شهادة", "خبير", "خبرة"],
    "حكم":          ["حكم", "قرار", "منطوق", "تعليل"],
    "طعن":          ["طعن", "تمييز", "استئناف", "اعتراض", "نقض"],
    "تنفيذ":        ["تنفيذ", "منفذ العدل", "حجز", "بيع بالمزاد"],
    "تحكيم":        ["تحكيم", "محكّم", "هيئة تحكيم"],
    "وقف_دعوى":     ["وقف", "تعليق", "إدخال", "انقطاع"],
    "تحفظي":        ["حجز تحفظي", "إجراء تحفظي", "وقتي", "مستعجل"],
}

def extract_topics(text: str) -> list:
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in text for kw in kws)]


def extract_osoul(input_path: Path) -> list:
    with open(input_path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    articles = []
    current_bab = current_fasl = current_qism = current_fara = ""
    current_article_num   = None
    current_article_lines = []

    def flush():
        if current_article_num is None:
            return
        text = "\n".join(current_article_lines).strip()
        if not text:
            return
        num_norm  = normalize_num(current_article_num).strip()
        is_mokrar = "مكرر" in num_norm
        num_clean = re.sub(r'\s*مكرر', '', num_norm).strip()
        try:
            num_int = int(num_clean)
        except ValueError:
            num_int = 0
        article_id = f"osoul_{num_clean}" + ("_مكرر" if is_mokrar else "")
        articles.append({
            "id": article_id,
            "article_number":     num_int,
            "article_number_str": f"المادة {num_clean}" + (" مكرر" if is_mokrar else ""),
            "text": text,
            "metadata": {
                "law_name": LAW_NAME, "law_year": LAW_YEAR, "type": LAW_TYPE,
                "bab": current_bab.strip(), "fasl": current_fasl.strip(),
                "qism": current_qism.strip(), "fara": current_fara.strip(),
                "topics": extract_topics(text),
            }
        })

    for line in lines:
        stripped = line.strip()
        if RE_BAB.match(stripped):
            current_bab = stripped; current_fasl = current_qism = current_fara = ""; continue
        if RE_FASL.match(stripped):
            current_fasl = stripped; current_qism = current_fara = ""; continue
        if RE_QISM.match(stripped):
            current_qism = stripped; current_fara = ""; continue
        if RE_FARA.match(stripped):
            current_fara = stripped; continue
        m = RE_ARTICLE.match(line)
        if m:
            flush()
            current_article_num = m.group(1)
            current_article_lines = []
            continue
        if current_article_num is not None:
            current_article_lines.append(line)

    flush()
    return articles


def main():
    print(f"[1] قراءة {INPUT_FILE} ...")
    articles = extract_osoul(INPUT_FILE)
    print(f"[2] تم استخراج {len(articles)} مادة")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[3] ✓ تم الحفظ: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
