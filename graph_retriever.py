#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_retriever.py
مسترجع الشبكة

الإصلاحات:
  1. لا Settings على مستوى الـ module (لا تعارض مع hybrid_rag)
  2. يقبل graph + payload_index جاهزين من الخارج (dependency injection)
     → لا scroll إضافي عند كل طلب
  3. graph node score = max(seed_scores_المتصلة) × DECAY بدلاً من 0.5 ثابتة
  4. from_qdrant() للاستخدام المستقل (اختبار مباشر)
  5. حُذفت _build_context و retrieve القديمتان — غير مستخدمتان

الاستخدام من hybrid_rag:
  gr = GraphRetriever(graph, payload_index)
  expanded = gr.expand(seed_items, depth=1)

الاستخدام المستقل:
  gr = GraphRetriever.from_qdrant(graph, qdrant_client)
  expanded = gr.expand(seed_items, depth=1)
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GRAPH_FILE  = Path(__file__).parent / "graph.json"
GRAPH_DECAY = 0.7   # score الجار = max(seed_scores المتصلة) × DECAY


def make_key(metadata: dict) -> str:
    """نفس منطق graph_builder"""
    source = metadata.get("source", "")
    if source == "fonon":
        return f"fonon_{metadata.get('template_id', '')}"
    return f"{source}_{metadata.get('article_num', '')}"


# ══════════════════════════════════════════════════════════════
class GraphRetriever:

    def __init__(self, graph: dict, payload_index: dict):
        """
        graph         : dict محمّل من graph.json
        payload_index : dict  مفتاح طبيعي → payload (من HybridRetriever)
        """
        self._graph         = graph
        self._payload_index = payload_index
        total = len(payload_index)
        print(f"[GraphRetriever] جاهز — {len(graph)} عقدة، {total} في الفهرس")

    # ── بناء standalone من Qdrant ─────────────────────────────
    @classmethod
    def from_qdrant(cls, graph: dict, qdrant_client, collections=None):
        """
        وضع standalone: يبني payload_index بنفسه.
        للاستخدام في الاختبار المباشر فقط.
        """
        if collections is None:
            collections = ("legal_laws", "legal_osoul", "legal_fonon")

        payload_index = {}
        for col in collections:
            offset = None
            while True:
                results, next_offset = qdrant_client.scroll(
                    collection_name=col,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in results:
                    key = make_key(pt.payload)
                    if key and key not in payload_index:
                        payload_index[key] = pt.payload
                if next_offset is None:
                    break
                offset = next_offset

        return cls(graph, payload_index)

    # ── استرجاع الجيران ───────────────────────────────────────
    def _get_neighbors(self, key: str, depth: int) -> set:
        visited  = {key}
        frontier = {key}
        for _ in range(depth):
            nxt = set()
            for k in frontier:
                for nb in self._graph.get(k, []):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
            frontier = nxt
        visited.discard(key)
        return visited

    # ── الدالة الرئيسية ───────────────────────────────────────
    def expand(self, seed_items: list, depth: int = 1) -> list:
        """
        يُوسّع قائمة seed_items بإضافة الجيران من الـ graph.

        seed_items : list of dicts { key, payload, score, source }
        المخرج     : seed_items + graph_items (بـ score ذكي)
        """
        seed_keys   = {item["key"] for item in seed_items}
        seed_scores = {item["key"]: item["score"] for item in seed_items}

        # جمع الجيران من كل الـ seeds
        neighbor_keys: set = set()
        for key in seed_keys:
            for nb in self._get_neighbors(key, depth):
                if nb not in seed_keys:
                    neighbor_keys.add(nb)

        graph_items = []
        for nb_key in neighbor_keys:
            payload = self._payload_index.get(nb_key)
            if not payload:
                continue

            # score = max(scores الـ seeds المتصلة) × DECAY
            connected = [
                seed_scores[sk]
                for sk in seed_keys
                if nb_key in self._graph.get(sk, [])
            ]
            score = round(max(connected) * GRAPH_DECAY if connected else 0.3, 4)

            graph_items.append({
                "key":     nb_key,
                "payload": payload,
                "score":   score,
                "source":  "graph",
            })

        return seed_items + graph_items


# ══════════════════════════════════════════════════════════════
# تشغيل مستقل للاختبار
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from qdrant_client import QdrantClient
    from llama_index.core import VectorStoreIndex
    from llama_index.core.settings import Settings
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.cohere import CohereEmbedding

    # الإعدادات هنا فقط — لا تأثير على الاستيراد من hybrid_rag
    Settings.embed_model = CohereEmbedding(
        api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-multilingual-v3.0",
        input_type="search_query",
    )
    Settings.llm = None

    if not GRAPH_FILE.exists():
        print("graph.json غير موجود — شغّل graph_builder.py أولاً")
        exit(1)

    with open(GRAPH_FILE, encoding="utf-8") as f:
        graph = json.load(f)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    gr = GraphRetriever.from_qdrant(graph, client)

    # اختبار مع seeds وهمية
    dummy_seeds = [
        {"key": "laws_137", "payload": gr._payload_index.get("laws_137", {}), "score": 0.9, "source": "dense"},
        {"key": "laws_138", "payload": gr._payload_index.get("laws_138", {}), "score": 0.75, "source": "bm25"},
    ]

    expanded = gr.expand(dummy_seeds, depth=1)
    graph_items = [x for x in expanded if x["source"] == "graph"]

    print(f"\n  Seed nodes  : {len(dummy_seeds)}")
    print(f"  Graph nodes : {len(graph_items)}")
    print(f"\n  جيران laws_137 و laws_138:")
    for item in sorted(graph_items, key=lambda x: -x["score"]):
        print(f"    {item['key']:20} score={item['score']:.3f}")