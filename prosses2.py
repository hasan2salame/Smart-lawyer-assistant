#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_osoul.py
أصول المحاكمات المدنية السورية — osoul.txt → JSON

الاستخدام:
    python 02_extract_osoul.py

المخرج:
    osoul_data.json  — مصفوفة من المواد
"""

import re
import json

# ── إعدادات ──────────────────────────────────────────────────────────────
INPUT_FILE  = r"data/osoul.txt"
OUTPUT_FILE = r"osoul_data.json"

LAW_NAME = "قانون أصول المحاكمات المدنية السوري"
LAW_YEAR = 2016
LAW_TYPE = "procedural"   # قانون إجرائي

# ── تطبيع الأرقام ────────────────────────────────────────────────────────
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_num(text: str) -> str:
    return text.translate(ARABIC_INDIC)

# ── أنماط ────────────────────────────────────────────────────────────────
# في أصول المحاكمات: " المادةN" (مسافة ثم المادة ثم الرقم بدون مسافة)
RE_ARTICLE = re.compile(
    r'^\s{0,4}المادة\s*([\d٠-٩]+(?:\s*مكرر)?)\s*$', re.U
)
RE_BAB   = re.compile(r'^\s*باب\s*.+|^باب\s*.+', re.U)
RE_FASL  = re.compile(r'^\s*الفصل\s+.+|^فصل\s*.+', re.U)
RE_QISM  = re.compile(r'^\s*القسم\s+.+', re.U)
RE_FARA  = re.compile(r'^\s*الفرع\s+.+', re.U)

# ── وسوم تلقائية ─────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "اختصاص":       ["اختصاص", "محكمة مختصة", "صلاحية"],
    "دعوى":         ["دعوى", "لائحة", "عريضة", "مدعي", "مدعى عليه"],
    "تبليغ":        ["تبليغ", "إخبار", "إعلام", "محضر تبليغ"],
    "جلسة":         ["جلسة", "موعد", "حضور", "غياب"],
    "إثبات":        ["إثبات", "بينة", "شاهد", "شهادة", "خبير", "خبرة"],
    "حكم":          ["حكم", "قرار", "اجتهاد", "منطوق", "تعليل"],
    "طعن":          ["طعن", "تمييز", "استئناف", "اعتراض", "نقض"],
    "تنفيذ":        ["تنفيذ", "منفذ العدل", "حجز", "بيع بالمزاد"],
    "تحكيم":        ["تحكيم", "محكّم", "هيئة تحكيم"],
    "نفقة_إجراءات": ["نفقة", "مؤنة"],
    "أحوال_شخصية":  ["أحوال شخصية", "زواج", "طلاق", "حضانة"],
    "وقف_دعوى":     ["وقف", "تعليق", "إدخال", "انقطاع"],
    "تحفظي":        ["حجز تحفظي", "إجراء تحفظي", "وقتي", "مستعجل"],
}

def extract_topics(text: str) -> list[str]:
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            topics.append(topic)
    return topics


# ══════════════════════════════════════════════════════════════════════════
def extract_osoul(input_path: str) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    articles = []
    current_bab  = ""
    current_fasl = ""
    current_qism = ""
    current_fara = ""

    current_article_num   = None
    current_article_lines = []

    # ──────────────────────────────────────────────────────────────────────
    def flush_article():
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

        article_id = f"osoul_{num_clean}"
        if is_mokrar:
            article_id += "_مكرر"

        doc = {
            "id":                 article_id,
            "article_number":     num_int,
            "article_number_str": f"المادة {num_clean}" + (" مكرر" if is_mokrar else ""),
            "text":               text,
            "metadata": {
                "law_name": LAW_NAME,
                "law_year": LAW_YEAR,
                "type":     LAW_TYPE,
                "bab":      current_bab.strip(),
                "fasl":     current_fasl.strip(),
                "qism":     current_qism.strip(),
                "fara":     current_fara.strip(),
                "topics":   extract_topics(text),
            }
        }
        articles.append(doc)

    # ──────────────────────────────────────────────────────────────────────
    for line in lines:
        stripped = line.strip()

        # تحديث التسلسل الهرمي
        if RE_BAB.match(stripped):
            current_bab  = stripped
            current_fasl = ""
            current_qism = ""
            current_fara = ""
            continue
        if RE_FASL.match(stripped):
            current_fasl = stripped
            current_qism = ""
            current_fara = ""
            continue
        if RE_QISM.match(stripped):
            current_qism = stripped
            current_fara = ""
            continue
        if RE_FARA.match(stripped):
            current_fara = stripped
            continue

        # رأس مادة جديدة؟
        m = RE_ARTICLE.match(line)   # نستخدم line (ليس stripped) للمحافظة على المسافة الأولى
        if m:
            flush_article()
            current_article_num   = m.group(1)
            current_article_lines = []
            continue

        if current_article_num is not None:
            current_article_lines.append(line)

    flush_article()
    return articles


# ══════════════════════════════════════════════════════════════════════════
def main():
    print(f"[1] قراءة {INPUT_FILE} ...")
    articles = extract_osoul(INPUT_FILE)

    print(f"[2] تم استخراج {len(articles)} مادة")

    topics_count = {}
    for art in articles:
        for t in art["metadata"]["topics"]:
            topics_count[t] = topics_count.get(t, 0) + 1
    print("    الموضوعات:")
    for k, v in sorted(topics_count.items(), key=lambda x: -x[1]):
        print(f"      {v:3d} × {k}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"[3] ✓ تم الحفظ: {OUTPUT_FILE}")

    print("\n── عيّنة أول 2 مواد ──")
    for art in articles[:2]:
        print(f"\n  id      : {art['id']}")
        print(f"  article : {art['article_number_str']}")
        print(f"  bab     : {art['metadata']['bab']}")
        print(f"  fasl    : {art['metadata']['fasl']}")
        print(f"  topics  : {art['metadata']['topics']}")
        print(f"  text    : {art['text'][:80]}...")


if __name__ == "__main__":
    main()