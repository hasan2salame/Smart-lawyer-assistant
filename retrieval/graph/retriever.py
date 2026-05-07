"""
retrieval/graph/retriever.py
Graph-based neighbour expansion for legal article retrieval.

Accepts a pre-built graph dict and the shared payload_index from
HybridRetriever, then expands a seed result list by adding connected
neighbour articles.

Neighbour score = max(connected seed scores) * GRAPH_DECAY
"""

from config import GRAPH_DECAY


def make_key(metadata: dict) -> str:
    """
    Build a natural string key from a Qdrant point's metadata.

    Examples
    --------
    laws_137   — source=laws,  article_num=137
    osoul_32   — source=osoul, article_num=32
    fonon_11   — source=fonon, template_id=11
    """
    source = metadata.get("source", "")
    if source == "fonon":
        return f"fonon_{metadata.get('template_id', '')}"
    return f"{source}_{metadata.get('article_num', '')}"


class GraphRetriever:
    """
    Expands retrieval results by walking the legal knowledge graph.

    Parameters
    ----------
    graph         : dict  — {key: [neighbour_key, ...]} adjacency list
    payload_index : dict  — {key: payload} shared from HybridRetriever
    """

    def __init__(self, graph: dict, payload_index: dict) -> None:
        self._graph         = graph
        self._payload_index = payload_index
        node_count = len(graph)
        edge_count = sum(len(v) for v in graph.values())
        print(f"[GraphRetriever] {node_count} nodes, {edge_count} edges")

    @classmethod
    def from_qdrant(
        cls,
        graph: dict,
        qdrant_client,
        collections: tuple = ("legal_laws", "legal_osoul", "legal_fonon"),
    ) -> "GraphRetriever":
        """
        Standalone constructor for testing — builds its own payload_index
        directly from Qdrant instead of sharing HybridRetriever's index.
        """
        payload_index: dict = {}
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

    def expand(self, seed_items: list[dict], depth: int = 1) -> list[dict]:
        """
        Expand seed_items by appending graph-adjacent articles.

        Parameters
        ----------
        seed_items : list[dict]  — {"key", "payload", "score", "source"}
        depth      : int         — BFS depth (default 1)

        Returns
        -------
        seed_items + neighbour items, each with source="graph" and a
        score derived from its highest-scoring connected seed.
        """
        seed_keys   = {item["key"] for item in seed_items}
        seed_scores = {item["key"]: item["score"] for item in seed_items}

        # BFS neighbour collection
        neighbour_keys: set[str] = set()
        for key in seed_keys:
            for nb in self._bfs_neighbours(key, depth):
                if nb not in seed_keys:
                    neighbour_keys.add(nb)

        graph_items: list[dict] = []
        for nb_key in neighbour_keys:
            payload = self._payload_index.get(nb_key)
            if not payload:
                continue

            # Score = max seed score of direct parents * decay
            connected_scores = [
                seed_scores[sk]
                for sk in seed_keys
                if nb_key in self._graph.get(sk, [])
            ]
            score = (
                round(max(connected_scores) * GRAPH_DECAY, 4)
                if connected_scores
                else 0.3
            )

            graph_items.append({
                "key":     nb_key,
                "payload": payload,
                "score":   score,
                "source":  "graph",
            })

        return seed_items + graph_items

    # ── Internal ──────────────────────────────────────────────────────────

    def _bfs_neighbours(self, key: str, depth: int) -> set[str]:
        """Return all keys reachable from *key* within *depth* hops."""
        visited  = {key}
        frontier = {key}
        for _ in range(depth):
            nxt: set[str] = set()
            for k in frontier:
                for nb in self._graph.get(k, []):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
            frontier = nxt
        visited.discard(key)
        return visited
