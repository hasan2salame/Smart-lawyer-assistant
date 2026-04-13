#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_graph.py
بناء شبكة العلاقات وحفظها في data/graph.json

يُشغَّل مرة واحدة بعد رفع البيانات لـ Qdrant،
أو عند تحديث البيانات.

الاستخدام:
    python scripts/build_graph.py
    # أو من Makefile:
    make build-graph
"""

import sys
import json
from pathlib import Path

# إضافة جذر المشروع لـ sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY, GRAPH_FILE
from retrieval.graph.builder import build_graph, report


def main():
    print("\n" + "=" * 50)
    print("  بناء شبكة العلاقات من Qdrant")
    print("=" * 50)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    graph = build_graph(client)

    # حفظ في data/graph.json
    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ تم الحفظ: {GRAPH_FILE}")
    report(graph)
    print("\n  جاهز — أعد تشغيل السيرفر لتحميل الـ graph الجديد\n")


if __name__ == "__main__":
    main()
