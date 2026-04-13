"""
retrieval/reranker.py
إعادة الترتيب بـ Cohere Rerank

يُطبَّق كمرحلة نهائية بعد RRF لاختيار أفضل النتائج
بناءً على الصلة الدلالية الدقيقة بالاستعلام.

ملاحظة fonon: العنوان يُضاف للنص لأن الكلمات المفتاحية
للقوالب تقع في العنوان وليس في نص الصياغة.
"""

import cohere

from config import COHERE_RERANK_MODEL, TOP_K_FINAL


class Reranker:
    """يُغلّف Cohere Rerank API."""

    def __init__(self, api_key: str):
        self._co = cohere.Client(api_key)

    def rerank(self, query: str, items: list, top_n: int = TOP_K_FINAL) -> list:
        """
        يُعيد ترتيب items بناءً على صلتها بـ query.

        المدخل:  list of dicts { key, payload, score, source }
        المخرج:  نفس الـ dicts مرتبة من جديد بـ relevance_score
        """
        if not items:
            return []

        docs = []
        for item in items:
            p    = item["payload"]
            text = p.get("original_text") or p.get("formal_text") or ""
            # القوالب: العنوان يُضاف لأن المصطلح القانوني فيه (خلعي، حضانة...)
            if p.get("source") == "fonon" and p.get("title"):
                text = p["title"] + "\n" + text
            docs.append(text)

        try:
            resp = self._co.rerank(
                query=query,
                documents=docs,
                top_n=min(top_n, len(docs)),
                model=COHERE_RERANK_MODEL,
            )
            reranked = []
            for r in resp.results:
                item = dict(items[r.index])
                item["score"] = round(r.relevance_score, 4)
                reranked.append(item)
            return reranked

        except Exception as e:
            print(f"  ⚠ Rerank: {e} — fallback للترتيب الأصلي")
            return items[:top_n]
