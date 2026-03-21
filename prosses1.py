#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_laws.py
قانون الأحوال الشخصية السوري — Laws.txt → JSON

الاستخدام:
    python 01_extract_laws.py

المخرج:
    laws_data.json  — مصفوفة من المواد، كل مادة وحدة واحدة للـ RAG
"""

import re
import json

# ── إعدادات ──────────────────────────────────────────────────────────────
INPUT_FILE  = r"data/syrian_personal_status_law.txt"
OUTPUT_FILE = r"laws_data.json"

LAW_NAME = "قانون الأحوال الشخصية السوري"
LAW_YEAR = 1953
LAW_TYPE = "substantive"   # قانون موضوعي (مقابل إجرائي)

# ── تطبيع الأرقام (عربية-هندية → عربية) ─────────────────────────────────
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_num(text: str) -> str:
    return text.translate(ARABIC_INDIC)

# ── أنماط التعرف على العناصر ─────────────────────────────────────────────
RE_KITAB = re.compile(r'^الكتاب\s+.+', re.U)
RE_BAB   = re.compile(r'^\s*الباب\s+.+', re.U)
RE_FASL  = re.compile(r'^\s*الفصل\s+.+', re.U)
RE_QISM  = re.compile(r'^\s*القسم\s+.+', re.U)

# مادة بالأرقام الغربية أو العربية-الهندية + مكرر اختياري
RE_ARTICLE = re.compile(
    r'^المادة\s+([\d٠-٩]+(?:\s*-\s*مكرر)?)\s*$', re.U
)

# ── توليد الوسوم التلقائية (topics) ──────────────────────────────────────
TOPIC_KEYWORDS = {
    "زواج":      ["زواج", "عقد الزواج", "خطبة", "مهر", "ولاية", "كفاءة"],
    "طلاق":      ["طلاق", "فسخ", "مخالعة", "تفريق", "عدة", "رجعة"],
    "نفقة":      ["نفقة", "مؤنة", "إعالة", "نفقات"],
    "حضانة":     ["حضانة", "رؤية", "مشاهدة"],
    "نسب":       ["نسب", "بنوة", "ولادة", "إقرار بالنسب"],
    "وصاية":     ["وصي", "وصاية", "قيّم", "قيمومة"],
    "ميراث":     ["ميراث", "إرث", "تركة", "وصية", "موروث"],
    "أهلية":     ["أهلية", "قاصر", "بلوغ", "رشد", "محجور"],
    "أحوال_مدنية": ["نفوس", "سجل", "وثيقة", "شهادة ولادة", "جنسية"],
}

def extract_topics(text: str) -> list[str]:
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            topics.append(topic)
    return topics


# ══════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية
# ══════════════════════════════════════════════════════════════════════════
def extract_laws(input_path: str) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    articles = []
    current_kitab = ""
    current_bab   = ""
    current_fasl  = ""
    current_qism  = ""

    current_article_num  = None
    current_article_lines = []

    # ──────────────────────────────────────────────────────────────────────
    def flush_article():
        """يحفظ المادة الحالية ويُضيفها للقائمة"""
        if current_article_num is None:
            return
        text = "\n".join(current_article_lines).strip()
        if not text:
            return

        # تطبيع رقم المادة
        num_norm = normalize_num(current_article_num).strip()

        # هل هي "مكرر"؟
        is_mokrar = "مكرر" in num_norm
        num_clean = re.sub(r'\s*-\s*مكرر', '', num_norm).strip()

        try:
            num_int = int(num_clean)
        except ValueError:
            num_int = 0

        article_id = f"law_احوال_{num_clean}"
        if is_mokrar:
            article_id += "_مكرر"

        doc = {
            "id":          article_id,
            "article_number":     num_int,
            "article_number_str": f"المادة {num_clean}" + (" - مكرر" if is_mokrar else ""),
            "text":        text,
            "metadata": {
                "law_name": LAW_NAME,
                "law_year": LAW_YEAR,
                "type":     LAW_TYPE,
                "kitab":    current_kitab.strip(),
                "bab":      current_bab.strip(),
                "fasl":     current_fasl.strip(),
                "qism":     current_qism.strip(),
                "topics":   extract_topics(text),
            }
        }
        articles.append(doc)

    # ──────────────────────────────────────────────────────────────────────
    # سكان خط بخط
    # ──────────────────────────────────────────────────────────────────────
    # الأسطر الأولى (قبل المادة 1) تحتوي الفهرست — نتجاهلها حتى أول مادة
    in_articles = False

    for line in lines:
        stripped = line.strip()

        # تحديث التسلسل الهرمي
        if RE_KITAB.match(stripped):
            current_kitab = stripped
            current_bab   = ""
            current_fasl  = ""
            current_qism  = ""
            continue
        if RE_BAB.match(stripped):
            current_bab  = stripped
            current_fasl = ""
            current_qism = ""
            continue
        if RE_FASL.match(stripped):
            current_fasl = stripped
            current_qism = ""
            continue
        if RE_QISM.match(stripped):
            current_qism = stripped
            continue

        # هل هذا السطر رأس مادة جديدة؟
        m = RE_ARTICLE.match(stripped)
        if m:
            flush_article()
            current_article_num   = m.group(1)
            current_article_lines = []
            in_articles = True
            continue

        # تجميع نص المادة الحالية
        if in_articles and current_article_num is not None:
            current_article_lines.append(line)

    flush_article()   # آخر مادة
    return articles


# ══════════════════════════════════════════════════════════════════════════
def main():
    print(f"[1] قراءة {INPUT_FILE} ...")
    articles = extract_laws(INPUT_FILE)

    print(f"[2] تم استخراج {len(articles)} مادة")

    # تقرير سريع
    topics_count = {}
    for art in articles:
        for t in art["metadata"]["topics"]:
            topics_count[t] = topics_count.get(t, 0) + 1
    print("    الموضوعات:")
    for k, v in sorted(topics_count.items(), key=lambda x: -x[1]):
        print(f"      {v:3d} × {k}")

    # حفظ JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"[3] ✓ تم الحفظ: {OUTPUT_FILE}")

    # عيّنة
    print("\n── عيّنة أول 2 مواد ──")
    for art in articles[:2]:
        print(f"\n  id           : {art['id']}")
        print(f"  article      : {art['article_number_str']}")
        print(f"  bab          : {art['metadata']['bab']}")
        print(f"  fasl         : {art['metadata']['fasl']}")
        print(f"  topics       : {art['metadata']['topics']}")
        print(f"  text (أول 80): {art['text'][:80]}...")


if __name__ == "__main__":
    main()