"""
retrieval/rrf.py
Reciprocal Rank Fusion  (Cormack et al., 2009).

Merges N independently ranked lists using only rank position — not raw
scores — which makes it robust to score distribution differences between
Dense and BM25 retrievers.

Formula:  RRF(d) = Σ  1 / (k + rank(d, list_i))
"""

from config import RRF_K, TOP_K_RRF


def rrf_merge(*ranked_lists: list[dict]) -> list[dict]:
    """
    Fuse an arbitrary number of ranked result lists via RRF.

    Each input list must contain dicts with at least {"key": str, "payload": dict}.
    Duplicate keys across lists are merged — the document with the highest
    accumulated RRF score wins the payload reference.

    Parameters
    ----------
    *ranked_lists : one or more lists of retrieval results

    Returns
    -------
    list[dict]  — up to TOP_K_RRF items, sorted by rrf_score descending.
                  Each item carries source="rrf" and the accumulated score.
    """
    rrf_scores: dict[str, float] = {}
    payloads:   dict[str, dict]  = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item.get("key")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            payloads.setdefault(key, item.get("payload", {}))

    top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RRF]

    return [
        {
            "key":       key,
            "payload":   payloads[key],
            "score":     score,
            "source":    "rrf",
        }
        for key, score in top
    ]
