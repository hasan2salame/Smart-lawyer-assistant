#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_graph.py
اختبار Graph Neighbor Expansion

يختبر توسيع نتائج البحث بالمواد المرتبطة
ويُظهر كيف يضيف الـ Graph سياقاً إضافياً.
النتائج في tests/results/graph_results.txt

الاستخدام:
    python tests/test_graph.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import HybridRetriever
from retrieval.rrf import rrf_merge

RESULTS_FILE = Path(__file__).parent / "results" / "graph_results.txt"

TEST_QUERIES = [
    "ما هي شروط الحضانة؟",
    "الطلاق الخلعي",
    "نفقة الأطفال",
]


def run():
    print("\n" + "═" * 55)
    print("  اختبار Graph Neighbor Expansion")
    print("═" * 55 + "\n")

    retriever = HybridRetriever()

    if not retriever._graph:
        print("  ⚠ graph.json غير موجود — شغّل: python scripts/build_graph.py")
        return

    lines = ["اختبار Graph Neighbor Expansion", "=" * 55, ""]

    for query in TEST_QUERIES:
        print(f"  السؤال: {query}")
        start = time.perf_counter()

        # بدون Graph
        d_laws  = retriever._dense.search(query, "legal_laws")
        d_osoul = retriever._dense.search(query, "legal_osoul")
        b_laws  = retriever._bm25.search(query,  "legal_laws")
        b_osoul = retriever._bm25.search(query,  "legal_osoul")
        rrf_items = rrf_merge(d_laws, d_osoul, b_laws, b_osoul)

        # مع Graph
        expanded = retriever._graph.expand(rrf_items, depth=1)
        elapsed  = round((time.perf_counter() - start) * 1000, 1)

        seed_keys  = {r["key"] for r in rrf_items}
        graph_only = [r for r in expanded if r["key"] not in seed_keys]

        lines.append(f"السؤال: {query}")
        lines.append(f"الوقت: {elapsed}ms")
        lines.append(f"Seeds (RRF): {len(rrf_items)} | Graph أضاف: {len(graph_only)}")
        lines.append("Seeds:")
        for r in rrf_items[:5]:
            p   = r["payload"]
            art = p.get("article_str") or p.get("title", r["key"])
            lines.append(f"  [seed  {r['score']:.4f}] {art}")
            print(f"    [seed  {r['score']:.4f}] {art}")

        lines.append("Graph nodes المضافة:")
        for r in sorted(graph_only, key=lambda x: -x["score"])[:5]:
            p   = r["payload"]
            art = p.get("article_str") or p.get("title", r["key"])
            lines.append(f"  [graph {r['score']:.4f}] {art}")
            print(f"    [graph {r['score']:.4f}] {art}")

        lines.append("")
        print()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ النتائج محفوظة: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
