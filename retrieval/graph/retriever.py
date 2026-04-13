"""
retrieval/graph/retriever.py
مسترجع شبكة العلاقات بين المواد القانونية

يقبل graph.json جاهزاً + payload_index مشتركاً من HybridRetriever،
ويُوسّع قائمة seed_items بإضافة المواد المرتبطة (الجيران).

score الجار = max(scores_seeds_المتصلة) × GRAPH_DECAY
"""

from config import GRAPH_DECAY


def make_key(metadata: dict) -> str:
    """
    يولّد مفتاحاً طبيعياً من metadata النقطة.

    أمثلة:
        laws_137   ← source=laws,  article_num=137
        osoul_32   ← source=osoul, article_num=32
        fonon_11   ← source=fonon, template_id=11
    """
    source = metadata.get("source", "")
    if source == "fonon":
        return f"fonon_{metadata.get('template_id', '')}"
    return f"{source}_{metadata.get('article_num', '')}"


class GraphRetriever:
    """
    يُوسّع نتائج البحث بالمواد المرتبطة عبر شبكة العلاقات.

    الاستخدام الأساسي (من HybridRetriever):
        gr = GraphRetriever(graph, payload_index)
        expanded = gr.expand(seed_items, depth=1)

    الاستخدام المستقل (للاختبار):
        gr = GraphRetriever.from_qdrant(graph, qdrant_client)
    """

    def __init__(self, graph: dict, payload_index: dict):
        self._graph         = graph
        self._payload_index = payload_index
        print(f"[GraphRetriever] جاهز — {len(graph)} عقدة، {len(payload_index)} في الفهرس")

    @classmethod
    def from_qdrant(cls, graph: dict, qdrant_client, collections=None):
        """
        وضع standalone: يبني payload_index من Qdrant بنفسه.
        يُستخدم للاختبار المباشر لـ graph_retriever.
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

    def _get_neighbors(self, key: str, depth: int) -> set:
        """يجمع جيران العقدة حتى العمق المحدد."""
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

    def expand(self, seed_items: list, depth: int = 1) -> list:
        """
        يُوسّع seed_items بإضافة الجيران من الـ graph.

        المدخل:  list of dicts { key, payload, score, source }
        المخرج:  seed_items + graph_items (مع score ذكي)
        """
        seed_keys   = {item["key"] for item in seed_items}
        seed_scores = {item["key"]: item["score"] for item in seed_items}

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
