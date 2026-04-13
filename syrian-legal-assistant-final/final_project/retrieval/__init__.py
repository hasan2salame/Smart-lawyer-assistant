"""
retrieval/__init__.py
HybridRetriever — المنسّق الرئيسي لطبقة الاسترجاع

يُهيّئ المكوّنات الأربعة مرة واحدة عند بدء التطبيق:
    DenseSearcher  → البحث الكثيف عبر Cohere + Qdrant
    BM25Searcher   → البحث المتفرق على كامل الـ corpus
    Reranker       → إعادة الترتيب عبر Cohere Rerank
    GraphRetriever → توسيع النتائج بالمواد المرتبطة

المسارات العامة:
    get_template(query)           → صياغة دعوى
    get_attachments(query)        → مرفقات مطلوبة
    answer_legal_question(query)  → إجابة قانونية

معمارية الاسترجاع (مسار 3):
    Dense(laws)  ──┐
    Dense(osoul) ──┤
                   ├─► RRF ─► Graph expand ─► Rerank ─► top-6
    BM25(laws)   ──┤
    BM25(osoul)  ──┘
"""

import json

from qdrant_client import QdrantClient
from llama_index.core.settings import Settings
from llama_index.embeddings.cohere import CohereEmbedding

from config import (
    COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    COHERE_EMBED_MODEL, GRAPH_FILE,
    COL_LAWS, COL_OSOUL, COL_FONON, TOP_K_FINAL,
)
from retrieval.dense    import DenseSearcher
from retrieval.bm25     import BM25Searcher
from retrieval.rrf      import rrf_merge
from retrieval.reranker import Reranker
from retrieval.graph.retriever import GraphRetriever, make_key


class HybridRetriever:
    """
    نقطة الدخول الوحيدة لطبقة الاسترجاع.
    يُنشأ كـ singleton في llm_layer ويُمرَّر لكل الـ handlers.
    """

    def __init__(self):
        print("[HybridRetriever] تهيئة...")

        Settings.embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name=COHERE_EMBED_MODEL,
            input_type="search_query",
        )
        Settings.llm = None

        self._qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )

        self._dense = DenseSearcher(
            qdrant_client=self._qdrant,
            collections=[COL_LAWS, COL_OSOUL, COL_FONON],
        )

        self._payload_index, corpus_keys = self._build_payload_index()

        self._bm25 = BM25Searcher(
            payload_index=self._payload_index,
            corpus_keys=corpus_keys,
        )

        self._reranker = Reranker(api_key=COHERE_API_KEY)

        self._graph: GraphRetriever | None = None
        if GRAPH_FILE.exists():
            with open(GRAPH_FILE, encoding="utf-8") as f:
                graph_data = json.load(f)
            self._graph = GraphRetriever(graph_data, self._payload_index)
            print(f"[HybridRetriever] Graph: {len(graph_data)} عقدة")
        else:
            print("[HybridRetriever] ⚠ graph.json غير موجود — شغّل: make build-graph")

        print("[HybridRetriever] ✅ جاهز\n")

    # ══════════════════════════════════════════════════════════
    # بناء الفهرس الداخلي
    # ══════════════════════════════════════════════════════════

    def _build_payload_index(self) -> tuple[dict, dict]:
        """Scroll كامل مرة واحدة → payload_index + corpus_keys"""
        payload_index = {}
        corpus_keys   = {}

        for col in (COL_LAWS, COL_OSOUL, COL_FONON):
            keys, offset = [], None
            while True:
                results, next_offset = self._qdrant.scroll(
                    collection_name=col, offset=offset,
                    limit=100, with_payload=True, with_vectors=False,
                )
                for pt in results:
                    key = make_key(pt.payload)
                    if key and key not in payload_index:
                        payload_index[key] = pt.payload
                        keys.append(key)
                if next_offset is None:
                    break
                offset = next_offset
            corpus_keys[col] = keys

        total = len(payload_index)
        print(f"[HybridRetriever] Payload index: {total} عقدة")
        for col in (COL_LAWS, COL_OSOUL, COL_FONON):
            print(f"  {col}: {len(corpus_keys.get(col, []))} وثيقة")

        return payload_index, corpus_keys

    # ══════════════════════════════════════════════════════════
    # مسار 1 — صياغة
    # ══════════════════════════════════════════════════════════

    def get_template(self, query: str) -> dict:
        """
        يُرجع القالب الأنسب.
        يُرسل كامل الـ 28 قالب لـ Rerank مباشرة (بدون Dense filter)
        لأن كلمات المفتاح في العنوان لا في نص الصياغة.
        """
        all_fonon = [
            {"key": k, "payload": p, "score": 1.0, "source": "fonon"}
            for k, p in self._payload_index.items()
            if p.get("source") == "fonon"
        ]
        if not all_fonon:
            return {"error": "لم يعثر على صياغة مناسبة"}

        reranked = self._reranker.rerank(query, all_fonon, top_n=1)
        payload  = reranked[0]["payload"]

        return {
            "title":       payload.get("title", ""),
            "formal_text": payload.get("formal_text", ""),
            "intro_notes": payload.get("intro_notes", ""),
            "post_notes":  payload.get("post_notes", ""),
            "category":    payload.get("category", ""),
            "attachments": [
                a.strip()
                for a in payload.get("attachments", "").split("|")
                if a.strip()
            ],
            "score": reranked[0]["score"],
        }

    # ══════════════════════════════════════════════════════════
    # مسار 2 — مرفقات
    # ══════════════════════════════════════════════════════════

    def get_attachments(self, query: str) -> dict:
        """يُرجع المرفقات المطلوبة للدعوى."""
        items = self._dense.search(query, COL_FONON)
        if not items:
            return {"error": "لم يعثر على دعوى مناسبة"}

        payload = items[0]["payload"]
        return {
            "title": payload.get("title", ""),
            "attachments": [
                a.strip()
                for a in payload.get("attachments", "").split("|")
                if a.strip()
            ],
            "category": payload.get("category", ""),
            "score":    items[0]["score"],
        }

    # ══════════════════════════════════════════════════════════
    # مسار 3 — سؤال قانوني
    # ══════════════════════════════════════════════════════════

    def answer_legal_question(self, query: str) -> dict:
        """
        True Hybrid RAG:
          [1] Dense + BM25 على laws و osoul (4 قوائم مستقلة)
          [2] RRF يدمج القوائم الأربع
          [3] Graph expand يضيف المواد المرتبطة
          [4] فلترة fonon
          [5] Rerank النهائي → top-6
        """
        d_laws  = self._dense.search(query, COL_LAWS)
        d_osoul = self._dense.search(query, COL_OSOUL)
        b_laws  = self._bm25.search(query, COL_LAWS)
        b_osoul = self._bm25.search(query, COL_OSOUL)

        rrf_items = rrf_merge(d_laws, d_osoul, b_laws, b_osoul)
        if not rrf_items:
            return {"error": "لم يعثر على مواد ذات صلة"}

        expanded = self._graph.expand(rrf_items, depth=1) if self._graph else rrf_items
        expanded = [i for i in expanded if i["payload"].get("source") != "fonon"]

        final = self._reranker.rerank(query, expanded, top_n=TOP_K_FINAL)
        return self._format_result(final)

    def _format_result(self, items: list) -> dict:
        articles, context_parts = [], []
        for item in items:
            payload     = item["payload"]
            text        = payload.get("original_text") or payload.get("formal_text", "")
            article_str = payload.get("article_str") or payload.get("title", item["key"])
            law_name    = payload.get("law_name", "")
            articles.append({
                "article": article_str,
                "law":     law_name,
                "text":    text,
                "score":   item["score"],
                "source":  item.get("source", ""),
            })
            context_parts.append(f"[{article_str} - {law_name}]\n{text}")
        return {
            "context":  "\n\n---\n\n".join(context_parts),
            "articles": articles,
        }
