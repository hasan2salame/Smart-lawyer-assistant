#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_rrf.py
اختبار Reciprocal Rank Fusion

يختبر دمج نتائج Dense + BM25 عبر RRF
ويقارن الترتيب قبل وبعد الدمج.
النتائج في tests/results/rrf_results.txt

الاستخدام:
    python tests/test_rrf.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import HybridRetriever
from retrieval.rrf import rrf_merge

RESULTS_FILE = Path(__file__).parent / "results" / "rrf_results.txt"

TEST_QUERIES = [
    "ما هي شروط الحضانة؟",
    "إجراءات تبليغ المدعى عليه",
    "شروط الطلاق الخلعي",
]


def run():
    print("\n" + "═" * 55)
    print("  اختبار RRF Fusion")
    print("═" * 55 + "\n")

    retriever = HybridRetriever()
    lines = ["اختبار Reciprocal Rank Fusion", "=" * 55, ""]

    for query in TEST_QUERIES:
        print(f"  السؤال: {query}")
        start = time.perf_counter()

        # 4 قوائم مستقلة
        d_laws  = retriever._dense.search(query, "legal_laws")
        d_osoul = retriever._dense.search(query, "legal_osoul")
        b_laws  = retriever._bm25.search(query,  "legal_laws")
        b_osoul = retriever._bm25.search(query,  "legal_osoul")

        merged  = rrf_merge(d_laws, d_osoul, b_laws, b_osoul)
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        # عناصر حصرية لكل مصدر
        dense_keys = {r["key"] for r in d_laws + d_osoul}
        bm25_keys  = {r["key"] for r in b_laws + b_osoul}
        both       = dense_keys & bm25_keys
        only_dense = dense_keys - bm25_keys
        only_bm25  = bm25_keys  - dense_keys

        lines.append(f"السؤال: {query}")
        lines.append(f"الوقت: {elapsed}ms")
        lines.append(f"Dense فقط: {len(only_dense)} | BM25 فقط: {len(only_bm25)} | كلاهما: {len(both)}")
        lines.append(f"بعد RRF (top-{len(merged)}):")

        for i, r in enumerate(merged[:8], 1):
            p      = r["payload"]
            art    = p.get("article_str") or p.get("title", r["key"])
            in_d   = "✓D" if r["key"] in dense_keys else "  "
            in_b   = "✓B" if r["key"] in bm25_keys  else "  "
            lines.append(f"  {i}. [RRF={r['score']:.6f}] {in_d}{in_b} {art}")
            print(f"    {i}. [RRF={r['score']:.6f}] {in_d}{in_b} {art}")

        lines.append("")
        print()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ النتائج محفوظة: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
