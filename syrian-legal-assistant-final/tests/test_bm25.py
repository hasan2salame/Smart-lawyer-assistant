#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_bm25.py
اختبار BM25 Sparse Search

يختبر البحث المتفرق على كامل الـ corpus
ويقارن نتائجه مع Dense لنفس الأسئلة.
النتائج في tests/results/bm25_results.txt

الاستخدام:
    python tests/test_bm25.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import HybridRetriever

RESULTS_FILE = Path(__file__).parent / "results" / "bm25_results.txt"

TEST_QUERIES = [
    "شروط الحضانة وأسباب سقوطها",
    "نفقة الزوجة",
    "تبليغ المدعى عليه",
    "الطلاق الخلعي المخالعة",
    "فسخ عقد الزواج",
    "المادة 137",   # اختبار بحث عن مادة محددة
    "استئناف الحكم",
]


def run():
    print("\n" + "═" * 55)
    print("  اختبار BM25 Sparse Search")
    print("═" * 55 + "\n")

    retriever = HybridRetriever()
    lines = ["اختبار BM25 Sparse Search", "=" * 55, ""]

    for query in TEST_QUERIES:
        print(f"  السؤال: {query}")
        start = time.perf_counter()

        res_laws  = retriever._bm25.search(query, "legal_laws")
        res_osoul = retriever._bm25.search(query, "legal_osoul")

        elapsed = round((time.perf_counter() - start) * 1000, 1)

        lines.append(f"السؤال: {query}")
        lines.append(f"الوقت: {elapsed}ms | laws: {len(res_laws)} | osoul: {len(res_osoul)}")
        lines.append("النتائج (top-5):")

        for i, r in enumerate((res_laws + res_osoul)[:5], 1):
            p   = r["payload"]
            art = p.get("article_str") or p.get("title", r["key"])
            law = p.get("law_name", "")[:30]
            lines.append(f"  {i}. [BM25={r['score']:.4f}] {art} — {law}")
            print(f"    {i}. [BM25={r['score']:.4f}] {art}")

        lines.append("")
        print()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ النتائج محفوظة: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
