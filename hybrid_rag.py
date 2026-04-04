#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_rag.py
محرك الاسترجاع الهجين الحقيقي

المسارات:
    get_template(query)           → صياغة
    get_attachments(query)        → مرفقات
    answer_legal_question(query)  → سؤال قانوني

معمارية True Hybrid RAG:

  Dense(laws)  ──┐
  Dense(osoul) ──┤
                 ├─► RRF ─► Graph expand ─► Rerank ─► top-6
  BM25(laws)   ──┤
  BM25(osoul)  ──┘

  الـ BM25 يعمل على كامل الـ corpus المبني مسبقاً عند التهيئة،
  وليس على نتائج Dense فقط — هذا هو الفرق الجوهري عن النسخة السابقة.

الإصلاحات:
  1. BM25 corpus كامل مبني عند التهيئة (مرة واحدة)
  2. RRF حقيقي يدمج 4 قوائم مستقلة
  3. GraphRetriever كـ singleton (لا إعادة بناء عند كل طلب)
  4. _normalize يعالج أ (U+0623) الناقصة
  5. نموذج بيانات موحد (dicts) طوال الـ pipeline
  6. حُذف: _rerank_dicts / _build_result_from_dicts / _bm25_ranks / _rrf (القديمة)
"""

import os
import json
import cohere
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding

from graph_retriever import GraphRetriever, make_key

load_dotenv()

COL_LAWS  = "legal_laws"
COL_OSOUL = "legal_osoul"
COL_FONON = "legal_fonon"

TOP_K_DENSE = 20   # نتائج Dense لكل collection
TOP_K_BM25  = 20   # نتائج BM25  لكل collection
TOP_K_RRF   = 15   # بعد دمج RRF
TOP_K_FINAL = 6    # بعد Rerank
RRF_K       = 60   # ثابت RRF القياسي

GRAPH_FILE = Path(__file__).parent / "graph.json"


# ══════════════════════════════════════════════════════════════
class HybridRetriever:

    def __init__(self):
        print("[HybridRetriever] تهيئة...")

        Settings.embed_model = CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-multilingual-v3.0",
            input_type="search_query",
        )
        Settings.llm = None

        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))

        # Dense Retrievers
        self._ret = {
            col: self._make_retriever(col)
            for col in (COL_LAWS, COL_OSOUL, COL_FONON)
        }

        # Scroll مرة واحدة → payload_index + BM25 corpus
        self._payload_index: dict = {}
        self._bm25: dict          = {}
        self._corpus_keys: dict   = {}
        self._build_indices()

        # Graph singleton
        self._graph_retriever: GraphRetriever | None = None
        if GRAPH_FILE.exists():
            with open(GRAPH_FILE, encoding="utf-8") as f:
                graph = json.load(f)
            # نمرر payload_index الجاهز — لا scroll إضافي
            self._graph_retriever = GraphRetriever(graph, self._payload_index)
            print(f"[HybridRetriever] Graph: {len(graph)} عقدة")
        else:
            print("[HybridRetriever] ⚠ graph.json غير موجود — بدون Graph")

        print("[HybridRetriever] ✅ جاهز\n")

    # ══════════════════════════════════════════════════════════
    # تهيئة داخلية
    # ══════════════════════════════════════════════════════════

    def _make_retriever(self, col: str):
        store = QdrantVectorStore(client=self.qdrant, collection_name=col)
        index = VectorStoreIndex.from_vector_store(store)
        return index.as_retriever(similarity_top_k=TOP_K_DENSE)

    def _build_indices(self):
        """
        Scroll كامل لكل collection مرة واحدة يبني:
          - self._payload_index : مفتاح → payload
          - self._bm25          : col → BM25Okapi على كامل الـ corpus
          - self._corpus_keys   : col → قائمة المفاتيح بنفس ترتيب BM25
        """
        for col in (COL_LAWS, COL_OSOUL, COL_FONON):
            keys, texts = [], []
            offset = None
            while True:
                results, next_offset = self.qdrant.scroll(
                    collection_name=col,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in results:
                    key = make_key(pt.payload)
                    if key and key not in self._payload_index:
                        self._payload_index[key] = pt.payload
                        keys.append(key)
                        text = (
                            pt.payload.get("original_text") or
                            pt.payload.get("formal_text") or ""
                        )
                        texts.append(text)
                if next_offset is None:
                    break
                offset = next_offset

            self._corpus_keys[col] = keys
            if texts:
                tokenized = [self._tokenize(t) for t in texts]
                self._bm25[col] = BM25Okapi(tokenized)
            else:
                self._bm25[col] = None

        total = len(self._payload_index)
        print(f"[HybridRetriever] Payload index: {total} عقدة")
        for col in (COL_LAWS, COL_OSOUL, COL_FONON):
            print(f"  {col}: {len(self._corpus_keys.get(col, []))} وثيقة")

    # ══════════════════════════════════════════════════════════
    # معالجة النص العربي
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(text: str) -> str:
        """تطبيع الحروف العربية للـ BM25 tokenization"""
        return (
            text.replace("أ", "ا")   # U+0623 → U+0627  كان ناقصاً
                .replace("إ", "ا")   # U+0625 → U+0627
                .replace("آ", "ا")   # U+0622 → U+0627
                .replace("ة", "ه")   # U+0629 → U+0647
                .replace("ى", "ي")   # U+0649 → U+064A
        )

    def _tokenize(self, text: str) -> list:
        return self._normalize(str(text)).split()

    # ══════════════════════════════════════════════════════════
    # دوال الاسترجاع الأساسية
    # ══════════════════════════════════════════════════════════

    def _dense_search(self, query: str, col: str) -> list:
        """Dense search → قائمة dicts موحدة"""
        try:
            nodes = self._ret[col].retrieve(query)
        except Exception as e:
            print(f"  ⚠ Dense {col}: {e}")
            return []
        result, seen = [], set()
        for n in nodes:
            key = make_key(n.node.metadata)
            if key and key not in seen:
                seen.add(key)
                result.append({
                    "key":     key,
                    "payload": n.node.metadata,
                    "score":   round(n.score or 0, 4),
                    "source":  "dense",
                })
        return result

    def _bm25_search(self, query: str, col: str) -> list:
        """
        True BM25 على كامل الـ corpus المبني مسبقاً
        → قائمة dicts موحدة مستقلة عن Dense
        """
        bm25 = self._bm25.get(col)
        keys = self._corpus_keys.get(col, [])
        if not bm25 or not keys:
            return []

        scores = bm25.get_scores(self._tokenize(query))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:TOP_K_BM25]

        result = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            key = keys[i]
            result.append({
                "key":     key,
                "payload": self._payload_index.get(key, {}),
                "score":   round(float(scores[i]), 4),
                "source":  "bm25",
            })
        return result

    def _rrf_merge(self, *ranked_lists: list) -> list:
        """
        True Reciprocal Rank Fusion من N قوائم مستقلة.
        كل قائمة تُعطي وزناً بالترتيب: 1 / (RRF_K + rank)
        العناصر الموجودة في أكثر من قائمة تتراكم أوزانها.
        """
        rrf_scores: dict[str, float] = {}
        all_items:  dict[str, dict]  = {}

        for ranked in ranked_lists:
            for rank, item in enumerate(ranked):
                key = item["key"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
                if key not in all_items:
                    all_items[key] = item

        merged = sorted(
            all_items.values(),
            key=lambda x: rrf_scores[x["key"]],
            reverse=True,
        )
        for item in merged:
            item["score"] = round(rrf_scores[item["key"]], 6)

        return merged[:TOP_K_RRF]

    def _rerank(self, query: str, items: list, top_n: int = TOP_K_FINAL) -> list:
        """Rerank موحد — يعمل على list of dicts مباشرةً"""
        if not items:
            return []
        docs = []
        for item in items:
            p    = item["payload"]
            text = p.get("original_text") or p.get("formal_text") or ""
            # fonon: العنوان حاسم للـ Rerank (الكلمات المفتاحية فيه لا في النص)
            if p.get("source") == "fonon" and p.get("title"):
                text = p["title"] + "\n" + text
            docs.append(text)
        try:
            resp = self.co.rerank(
                query=query,
                documents=docs,
                top_n=min(top_n, len(docs)),
                model="rerank-multilingual-v3.0",
            )
            reranked = []
            for r in resp.results:
                item = dict(items[r.index])
                item["score"] = round(r.relevance_score, 4)
                reranked.append(item)
            return reranked
        except Exception as e:
            print(f"  ⚠ Rerank: {e}")
            return items[:top_n]

    def _format_result(self, items: list) -> dict:
        """تحويل list of dicts إلى المخرج النهائي"""
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

    # ══════════════════════════════════════════════════════════
    # المسارات العامة (Public API)
    # ══════════════════════════════════════════════════════════

    def get_template(self, query: str) -> dict:
        """
        مسار 1 — صياغة
        Fonon: 28 وثيقة فقط → Rerank مباشرة على كامل الـ corpus
        لا نستخدم Dense كفلتر هنا لأن:
          1. 28 وثيقة فقط → Rerank سريع على الكل
          2. BM25 لا يحل مشكلة الصرف العربي (خلعي ≠ الخلعي)
          3. Dense قد يفوّت الوثيقة الصحيحة في top-20
        → إرسال كامل الـ corpus للـ Rerank يضمن النتيجة الصحيحة دائماً
        """
        # كل الـ 28 قالب من payload_index مباشرة
        all_fonon = [
            {"key": k, "payload": p, "score": 1.0, "source": "fonon"}
            for k, p in self._payload_index.items()
            if p.get("source") == "fonon"
        ]
        if not all_fonon:
            return {"error": "لم يعثر على صياغة مناسبة"}

        reranked = self._rerank(query, all_fonon, top_n=1)
        best     = reranked[0]
        payload  = best["payload"]

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
            "score": best["score"],
        }

    def get_attachments(self, query: str) -> dict:
        """
        مسار 2 — مرفقات
        Fonon: Dense → top-1 (سريع، لا حاجة لـ Rerank)
        """
        items = self._dense_search(query, COL_FONON)
        if not items:
            return {"error": "لم يعثر على دعوى مناسبة"}

        best    = items[0]
        payload = best["payload"]

        return {
            "title": payload.get("title", ""),
            "attachments": [
                a.strip()
                for a in payload.get("attachments", "").split("|")
                if a.strip()
            ],
            "category": payload.get("category", ""),
            "score":    best["score"],
        }

    def answer_legal_question(self, query: str) -> dict:
        """
        مسار 3 — سؤال قانوني

        True Hybrid RAG:
          Dense(laws)  ─┐
          Dense(osoul) ─┤
                        ├─► RRF ─► Graph expand ─► Rerank ─► top-6
          BM25(laws)   ─┤
          BM25(osoul)  ─┘
        """
        # ── 1. استرجاع مستقل (4 قوائم) ───────────────────────
        d_laws  = self._dense_search(query, COL_LAWS)
        d_osoul = self._dense_search(query, COL_OSOUL)
        b_laws  = self._bm25_search(query, COL_LAWS)
        b_osoul = self._bm25_search(query, COL_OSOUL)

        # ── 2. True RRF من 4 قوائم مستقلة ────────────────────
        rrf_items = self._rrf_merge(d_laws, d_osoul, b_laws, b_osoul)
        if not rrf_items:
            return {"error": "لم يعثر على مواد ذات صلة"}

        # ── 3. توسيع بالـ Graph (singleton جاهز) ──────────────
        if self._graph_retriever:
            expanded = self._graph_retriever.expand(rrf_items, depth=1)
        else:
            expanded = rrf_items

        # ── 4. فلترة fonon — لا قوالب في إجابة السؤال القانوني
        expanded = [i for i in expanded if i["payload"].get("source") != "fonon"]

        # ── 5. Rerank النهائي ──────────────────────────────────
        final = self._rerank(query, expanded, top_n=TOP_K_FINAL)

        return self._format_result(final)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SEP = "═" * 60
    r   = HybridRetriever()

    print(f"\n{SEP}\n  مسار 1 — صياغة\n{SEP}")
    res = r.get_template("دعوى طلاق خلعي")
    print(f"  العنوان: {res.get('title')}")
    print(f"  Score  : {res.get('score')}")

    print(f"\n{SEP}\n  مسار 2 — مرفقات\n{SEP}")
    res = r.get_attachments("دعوى نفقة")
    print(f"  العنوان  : {res.get('title')}")
    print(f"  المرفقات : {res.get('attachments')}")

    print(f"\n{SEP}\n  مسار 3 — سؤال قانوني\n{SEP}")
    res = r.answer_legal_question("ما هي شروط الحضانة؟")
    for a in res.get("articles", []):
        print(f"  [{a.get('source',''):6}] {a['article']:30} score={a['score']:.4f}")