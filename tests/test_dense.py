#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_dense.py
اختبار Dense Search (Cohere Embeddings + Qdrant)

يختبر جودة البحث الكثيف بأسئلة قانونية حقيقية
ويحفظ النتائج في tests/results/dense_results.txt

الاستخدام:
    python tests/test_dense.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import HybridRetriever

RESULTS_FILE = Path(__file__).parent / "results" / "dense_results.txt"

TEST_QUERIES = [
    "ما هي شروط الحضانة؟",
    "متى تسقط نفقة الزوجة؟",
    "إجراءات تبليغ المدعى عليه",
    "شروط الطلاق الخلعي",
    "أسباب فسخ عقد الزواج",
]


def run():
    print("\n" + "═" * 55)
    print("  اختبار Dense Search")
    print("═" * 55 + "\n")

    retriever = HybridRetriever()
    lines = ["اختبار Dense Search", "=" * 55, ""]

    for query in TEST_QUERIES:
        print(f"  السؤال: {query}")
        start = time.perf_counter()

        results_laws  = retriever._dense.search(query, "legal_laws")
        results_osoul = retriever._dense.search(query, "legal_osoul")

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        all_results = results_laws + results_osoul

        lines.append(f"السؤال: {query}")
        lines.append(f"الوقت: {elapsed}ms | laws: {len(results_laws)} | osoul: {len(results_osoul)}")
        lines.append("النتائج (top-5):")

        for i, r in enumerate(all_results[:5], 1):
            p = r["payload"]
            art = p.get("article_str") or p.get("title", r["key"])
            law = p.get("law_name", "")[:30]
            lines.append(f"  {i}. [{r['score']:.4f}] {art} — {law}")
            print(f"    {i}. [{r['score']:.4f}] {art}")

        lines.append("")
        print()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ النتائج محفوظة: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
