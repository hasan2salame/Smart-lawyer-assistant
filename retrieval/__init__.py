"""
retrieval/__init__.py
HybridRetriever — single entry point for the entire retrieval layer.
"""

import re
import json

from qdrant_client import QdrantClient
from llama_index.core.settings import Settings
from llama_index.embeddings.cohere import CohereEmbedding

from config import (
    COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    COHERE_EMBED_MODEL, GRAPH_FILE,
    COL_LAWS, COL_OSOUL, COL_FONON,
    TOP_K_FINAL, GRAPH_DECAY, ADAPTIVE_K_MAX,
)
from retrieval.dense            import DenseSearcher
from retrieval.bm25             import BM25Searcher
from retrieval.rrf              import rrf_merge
from retrieval.reranker         import Reranker
from retrieval.adaptive_k       import adaptive_filter
from retrieval.query_processor  import clean_text, rewrite_query
from retrieval.graph.retriever  import GraphRetriever, make_key

# Pattern for direct article lookup: "المادة 137" / "م/95" / "نص المادة 85"
_ARTICLE_NUM_RE = re.compile(
    r"(?:ال|ل|ب)?مادة\s*(\d+)|م\s*[./]\s*(\d+)|نص\s+المادة\s+(\d+)",
    re.UNICODE,
)


class HybridRetriever:
    """
    Singleton orchestrator for the retrieval layer.
    Instantiated once in llm.py and shared across all request handlers.
    """

    def __init__(self) -> None:
        print("[HybridRetriever] Initialising...")

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
            collections=[COL_LAWS, COL_OSOUL],
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
        else:
            print("[HybridRetriever] WARNING: graph.json not found. "
                  "Run: python scripts/build_graph.py")

        print("[HybridRetriever] Ready\n")

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_payload_index(self) -> tuple[dict, dict]:
        payload_index: dict = {}
        corpus_keys:   dict = {}

        for col in (COL_LAWS, COL_OSOUL, COL_FONON):
            keys, offset = [], None
            while True:
                results, next_offset = self._qdrant.scroll(
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
                        keys.append(key)
                if next_offset is None:
                    break
                offset = next_offset
            corpus_keys[col] = keys

        print(f"[HybridRetriever] Payload index: {len(payload_index)} nodes")
        for col in (COL_LAWS, COL_OSOUL, COL_FONON):
            print(f"  {col}: {len(corpus_keys.get(col, []))} documents")

        return payload_index, corpus_keys

    # ── Path 1 — Template drafting ─────────────────────────────────────

    def get_template(self, query: str) -> dict:
        clean_query = clean_text(query)
        all_fonon = [
            {"key": k, "payload": p, "score": 1.0, "source": "fonon"}
            for k, p in self._payload_index.items()
            if p.get("source") == "fonon"
        ]
        if not all_fonon:
            return {"error": "No template found"}

        reranked = self._reranker.rerank(clean_query, all_fonon, top_n=1)
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

    # ── Path 2 — Required attachments ─────────────────────────────────

    def get_attachments(self, query: str) -> dict:
        clean_query = clean_text(query)
        all_fonon = [
            {"key": k, "payload": p, "score": 1.0, "source": "fonon"}
            for k, p in self._payload_index.items()
            if p.get("source") == "fonon"
        ]
        if not all_fonon:
            return {"error": "No matching case type found"}

        reranked = self._reranker.rerank(clean_query, all_fonon, top_n=1)
        payload  = reranked[0]["payload"]

        return {
            "title": payload.get("title", ""),
            "attachments": [
                a.strip()
                for a in payload.get("attachments", "").split("|")
                if a.strip()
            ],
            "category": payload.get("category", ""),
            "score":    reranked[0]["score"],
        }

    # ── Path 3 — Legal Q&A (True Hybrid RAG) ──────────────────────────

    def answer_legal_question(self, query: str) -> dict:
        """
        Full Hybrid RAG pipeline with pure gap-based adaptive K.

        Steps
        -----
        [0] Direct article lookup — "المادة X" → immediate return
        [1] clean + rewrite query to formal MSA
        [2] Four independent retrieval lists (Dense × 2, BM25 × 2)
        [3] RRF fusion across all four lists
        [4] Graph neighbour expansion (depth=1, decay=GRAPH_DECAY)
        [5] Remove fonon documents from legal QA context
        [6] Cohere Rerank (TOP_K_FINAL=25 pool)
        [7] Pure gap-based adaptive K — no fixed minimum
        [8] Graph guaranteed inclusion — top-3 seed neighbours appended
        [9] Out-of-scope guard — empty result → error
        """
        # [0] Direct lookup for explicit article references
        m = _ARTICLE_NUM_RE.search(query)
        if m:
            num = int(next(g for g in m.groups() if g is not None))
            for src in ("laws", "osoul"):
                key = f"{src}_{num}"
                if key in self._payload_index:
                    return self._format_result([{
                        "key":     key,
                        "payload": self._payload_index[key],
                        "score":   1.0,
                        "source":  "direct",
                    }])

        # [1] Preprocess
        clean_q  = clean_text(query)
        formal_q = rewrite_query(clean_q)

        # [2] Four independent retrieval lists
        d_laws  = self._dense.search(formal_q, COL_LAWS)
        d_osoul = self._dense.search(formal_q, COL_OSOUL)
        b_laws  = self._bm25.search(formal_q,  COL_LAWS)
        b_osoul = self._bm25.search(formal_q,  COL_OSOUL)

        # [3] RRF fusion
        rrf_items = rrf_merge(d_laws, d_osoul, b_laws, b_osoul)
        if not rrf_items:
            return {"error": "out_of_scope"}

        # [4] Graph expansion
        expanded = (
            self._graph.expand(rrf_items, depth=1)
            if self._graph else rrf_items
        )

        # [5] Strip fonon documents
        expanded = [i for i in expanded if i["payload"].get("source") != "fonon"]

        # [6] Rerank — large pool (TOP_K_FINAL=25) for gap detection
        reranked = self._reranker.rerank(formal_q, expanded, top_n=TOP_K_FINAL)

        # [7] Pure gap-based adaptive K
        final = adaptive_filter(reranked)

        # [8] Graph guaranteed inclusion
        # بعد الـ adaptive filter نضيف جيران مباشرين للـ seeds المختارة
        # هؤلاء لا يعيدون المرور على الـ reranker — يُضافون مباشرة
        if self._graph and len(final) < ADAPTIVE_K_MAX:
            seen = {item["key"] for item in final}
            for seed in final[:3]:  # جيران أهم 3 seeds
                for nb_key in self._graph._graph.get(seed["key"], [])[:2]:
                    if nb_key not in seen:
                        payload = self._payload_index.get(nb_key)
                        if payload and payload.get("source") != "fonon":
                            final.append({
                                "key":     nb_key,
                                "payload": payload,
                                "score":   round(seed["score"] * GRAPH_DECAY, 4),
                                "source":  "graph_guaranteed",
                            })
                            seen.add(nb_key)
                    if len(final) >= ADAPTIVE_K_MAX:
                        break

        # [9] Out-of-scope guard
        if not final:
            return {"error": "out_of_scope"}

        # Clean text before sending to LLM
        clean_final = []
        for item in final:
            payload    = item["payload"]
            clean_item = {**item, "payload": {**payload}}
            raw_text   = payload.get("original_text", "")
            if raw_text:
                clean_item["payload"]["original_text"] = clean_text(raw_text)
            clean_final.append(clean_item)

        return self._format_result(clean_final)

    # ── Shared helpers ─────────────────────────────────────────────────

    def _format_result(self, items: list[dict]) -> dict:
        articles:      list[dict] = []
        context_parts: list[str]  = []

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
