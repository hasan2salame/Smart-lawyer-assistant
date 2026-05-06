"""
retrieval/adaptive_k.py
Gap-based dynamic Top-K selection (elbow method).

After Cohere Rerank the score distribution often has a natural "cliff" —
a large drop that separates relevant results from noise.  This module finds
that cliff and returns only the items above it instead of a fixed Top-K.

Design decision — no absolute score threshold
---------------------------------------------
Cohere rerank-multilingual-v3.0 scores are RELATIVE, not absolute probabilities.
A score of 0.05 may mean "this is the best available article for this query" —
not that the article is irrelevant.  Using a fixed threshold on absolute scores
produces false negatives for valid legal questions in a specialised Arabic corpus.

The correct out-of-scope guard is:
  - The intent classifier blocks non-legal queries before they reach retrieval.
  - If retrieval returns an empty list → out_of_scope.
  - If retrieval returns ANY articles → proceed and let the LLM report if the
    context is insufficient.
"""

from config import ADAPTIVE_K_MIN, ADAPTIVE_K_MAX


def adaptive_filter(reranked: list[dict]) -> list[dict]:
    """
    Return the subset of reranked items that sit above the largest score gap.

    Steps
    -----
    1. Min-Max normalise all scores to [0, 1].
    2. Find the index with the largest consecutive drop (the "elbow").
    3. Clamp the cut-point to [ADAPTIVE_K_MIN, ADAPTIVE_K_MAX].
    4. Return items[:cut].

    Returns an empty list only if the input list is empty — never based on
    an absolute score threshold (see module docstring for rationale).
    """
    if not reranked:
        return []

    scores = [item["score"] for item in reranked]

    # Single result — return it unconditionally
    if len(scores) == 1:
        return reranked

    lo, hi = min(scores), max(scores)
    if hi == lo:
        return reranked[:ADAPTIVE_K_MIN]

    normed = [(s - lo) / (hi - lo) for s in scores]

    # Find the largest consecutive gap (the "elbow")
    best_gap, best_idx = 0.0, 0
    for i in range(len(normed) - 1):
        gap = normed[i] - normed[i + 1]
        if gap > best_gap:
            best_gap, best_idx = gap, i

    cut = best_idx + 1
    cut = max(ADAPTIVE_K_MIN, min(cut, ADAPTIVE_K_MAX))
    return reranked[:cut]


def check_confidence(reranked: list[dict]) -> bool:
    """Return True if the reranked list is non-empty (retrieval found something)."""
    return bool(reranked)
