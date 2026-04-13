"""
retrieval/rrf.py
Reciprocal Rank Fusion — دمج قوائم مستقلة

خوارزمية RRF (Cormack & Clarke, 2009):
    score(d) = Σ  1 / (k + rank(d, list_i))

تقبل N قوائم مرتبة وتُرجع قائمة واحدة مدموجة.
العناصر الموجودة في أكثر من قائمة تتراكم أوزانها تلقائياً.
"""

from config import RRF_K, TOP_K_RRF


def rrf_merge(*ranked_lists: list, top_k: int = TOP_K_RRF) -> list:
    """
    يدمج N قوائم مستقلة بخوارزمية RRF.

    المدخلات:
        *ranked_lists : قوائم مرتبة، كل عنصر dict يحتوي "key"
        top_k         : عدد النتائج المُرجَعة

    المخرج: list of dicts مرتبة تنازلياً بـ RRF score
        كل dict يحتوي "score" مُحدَّث بقيمة RRF
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

    # تحديث الـ score بقيمة RRF النهائية
    for item in merged:
        item["score"] = round(rrf_scores[item["key"]], 6)

    return merged[:top_k]
