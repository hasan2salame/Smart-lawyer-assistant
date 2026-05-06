"""
tests/test_dense.py
Integration tests for DenseSearcher — requires live Qdrant + Cohere.

Run only when both services are available:
    python tests/test_dense.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from llama_index.core.settings import Settings
from llama_index.embeddings.cohere import CohereEmbedding

from config import (
    COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    COHERE_EMBED_MODEL, COL_LAWS, COL_OSOUL, TOP_K_DENSE,
)
from retrieval.dense import DenseSearcher

TEST_QUERIES = [
    ("شروط الحضانة وانتقالها",        COL_LAWS),
    ("الطلاق الخلعي وإجراءاته",        COL_LAWS),
    ("تبليغ المدعى عليه بالدعوى",      COL_OSOUL),
    ("إجراءات الاستئناف أمام المحكمة", COL_OSOUL),
    ("نفقة الزوجة بعد الطلاق",         COL_LAWS),
]


if __name__ == "__main__":
    Settings.embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name=COHERE_EMBED_MODEL,
        input_type="search_query",
    )
    Settings.llm = None

    client   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    searcher = DenseSearcher(qdrant_client=client, collections=[COL_LAWS, COL_OSOUL])

    SEP = "─" * 70
    print(f"\nDenseSearcher Integration Test — {len(TEST_QUERIES)} queries\n{SEP}")

    for query, col in TEST_QUERIES:
        t0      = time.perf_counter()
        results = searcher.search(query, col)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        icon = "✅" if results else "❌"
        print(f"\n{icon}  [{col}] {query}  ({elapsed} ms, {len(results)} results)")
        for r in results[:3]:
            p = r["payload"]
            print(f"     {r['score']:.4f}  {p.get('article_str', r['key'])} "
                  f"— {p.get('law_name', '')}")

    print(f"\n{SEP}")
