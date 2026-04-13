"""
retrieval/dense.py
البحث الكثيف (Dense Search) عبر Cohere Embeddings + Qdrant

يُنشئ retriever لكل collection ويُرجع نتائج موحدة كـ list of dicts.
"""

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import TOP_K_DENSE
from retrieval.graph.retriever import make_key


class DenseSearcher:
    """
    يُغلّف LlamaIndex retrievers لكل collection.
    يُهيَّأ مرة واحدة ويُستخدم طوال عمر التطبيق.
    """

    def __init__(self, qdrant_client, collections: list):
        """
        qdrant_client : QdrantClient جاهز
        collections   : قائمة بأسماء الـ collections
        """
        self._retrievers = {}
        for col in collections:
            store = QdrantVectorStore(client=qdrant_client, collection_name=col)
            index = VectorStoreIndex.from_vector_store(store)
            self._retrievers[col] = index.as_retriever(similarity_top_k=TOP_K_DENSE)

    def search(self, query: str, collection: str) -> list:
        """
        يُجري Dense search على collection محددة.

        المخرج: list of dicts
            { key, payload, score, source="dense" }
        """
        retriever = self._retrievers.get(collection)
        if not retriever:
            return []

        try:
            nodes = retriever.retrieve(query)
        except Exception as e:
            print(f"  ⚠ Dense [{collection}]: {e}")
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
