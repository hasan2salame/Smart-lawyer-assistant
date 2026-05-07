"""
retrieval/dense.py
Semantic (dense) retrieval via Cohere embeddings + Qdrant vector store,
bridged through LlamaIndex VectorStoreIndex.
"""

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import COL_LAWS, COL_OSOUL, COL_FONON, TOP_K_DENSE
from retrieval.graph.retriever import make_key


class DenseSearcher:
    """
    Builds a LlamaIndex VectorStoreIndex for each requested collection
    at startup, then exposes a single search() method.

    Parameters
    ----------
    qdrant_client : QdrantClient   — shared client instance
    collections   : list[str]      — collections to index
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collections: list[str],
    ) -> None:
        self._client  = qdrant_client
        self._indices: dict[str, VectorStoreIndex] = {}

        for col in collections:
            store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=col,
            )
            self._indices[col] = VectorStoreIndex.from_vector_store(store)

        print(f"[DenseSearcher] Indexed collections: {collections}")

    def search(self, query: str, collection: str) -> list[dict]:
        """
        Return up to TOP_K_DENSE results from *collection* ranked by
        cosine similarity.

        Returns an empty list if the collection was not indexed or the
        query string is empty.
        """
        if not query or collection not in self._indices:
            return []

        retriever = self._indices[collection].as_retriever(
            similarity_top_k=TOP_K_DENSE
        )
        nodes = retriever.retrieve(query)

        return [
            {
                "key":     make_key(node.metadata) or node.node_id,
                "payload": node.metadata,
                "score":   node.score or 0.0,
                "source":  "dense",
            }
            for node in nodes
        ]
