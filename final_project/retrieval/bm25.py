"""
retrieval/bm25.py
البحث المتفرق (Sparse Search) عبر BM25 على كامل الـ corpus

يُبنى الـ corpus عند التهيئة مرة واحدة بـ scroll كامل من Qdrant،
وليس على نتائج Dense فقط — هذا هو الفرق عن الأساليب المبسّطة.

التطبيع العربي: يوحّد أشكال الألف والتاء المربوطة والياء
لتحسين دقة المطابقة في النصوص القانونية العربية.
"""

from rank_bm25 import BM25Okapi

from config import TOP_K_BM25
from retrieval.graph.retriever import make_key


class BM25Searcher:
    """
    يُجري BM25 search على corpus كامل مبني مسبقاً.
    يُهيَّأ مرة واحدة من payload_index الجاهز.
    """

    def __init__(self, payload_index: dict, corpus_keys: dict):
        """
        payload_index : dict  مفتاح طبيعي → payload  (من HybridRetriever)
        corpus_keys   : dict  collection → [keys بترتيب الـ corpus]
        """
        self._payload_index = payload_index
        self._corpus_keys   = corpus_keys
        self._bm25: dict    = {}

        # بناء BM25Okapi لكل collection
        for col, keys in corpus_keys.items():
            texts = [
                payload_index.get(k, {}).get("original_text") or
                payload_index.get(k, {}).get("formal_text") or ""
                for k in keys
            ]
            if texts:
                self._bm25[col] = BM25Okapi([self._tokenize(t) for t in texts])

    # ── تطبيع عربي ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """توحيد أشكال الحروف العربية لتحسين المطابقة"""
        return (
            text.replace("أ", "ا")  # U+0623
                .replace("إ", "ا")  # U+0625
                .replace("آ", "ا")  # U+0622
                .replace("ة", "ه")  # U+0629
                .replace("ى", "ي")  # U+0649
        )

    def _tokenize(self, text: str) -> list:
        return self._normalize(str(text)).split()

    # ── البحث ─────────────────────────────────────────────────────────────

    def search(self, query: str, collection: str) -> list:
        """
        يُجري BM25 search على collection محددة.

        المخرج: list of dicts
            { key, payload, score, source="bm25" }
        """
        bm25 = self._bm25.get(collection)
        keys = self._corpus_keys.get(collection, [])
        if not bm25 or not keys:
            return []

        scores = bm25.get_scores(self._tokenize(query))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
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
