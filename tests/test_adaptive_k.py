"""
tests/test_adaptive_k.py
Unit tests and visual demonstration for adaptive_filter() and check_confidence().

Two sections
------------
[1] Unit tests  — correctness assertions  (run silently in CI)
[2] Visual demo — shows the elbow method in action across 5 score distributions,
                  proving the system uses dynamic K, not a fixed top-K.

No external APIs required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.adaptive_k import adaptive_filter, check_confidence
from config import CONFIDENCE_THRESHOLD, ADAPTIVE_K_MIN, ADAPTIVE_K_MAX


def _items(scores: list[float]) -> list[dict]:
    return [{"key": f"doc_{i}", "payload": {"article_str": f"Article {i+1}"},
             "score": s} for i, s in enumerate(scores)]


# ── Section 1 — Unit tests ────────────────────────────────────────────────

def test_empty_input_returns_empty():
    assert adaptive_filter([]) == []


def test_below_threshold_no_longer_rejects():
    # Threshold-based rejection was removed — a single retrieved article is
    # always returned regardless of its absolute score.
    # Out-of-scope is handled by the classifier and empty retrieval list.
    result = adaptive_filter(_items([CONFIDENCE_THRESHOLD - 0.01]))
    assert len(result) == 1


def test_single_item_above_threshold():
    result = adaptive_filter(_items([CONFIDENCE_THRESHOLD + 0.1]))
    assert len(result) == 1


def test_all_equal_scores_returns_k_min():
    result = adaptive_filter(_items([0.8, 0.8, 0.8, 0.8]))
    assert len(result) == ADAPTIVE_K_MIN


def test_clear_cliff_at_position_2():
    # Large gap after index 1 — only first 2 items are above the cliff
    result = adaptive_filter(_items([0.9, 0.85, 0.3, 0.25, 0.2]))
    assert len(result) == 2


def test_clear_cliff_at_position_3():
    result = adaptive_filter(_items([0.9, 0.88, 0.85, 0.3, 0.28, 0.1]))
    assert len(result) == 3


def test_cliff_position_determines_k():
    # Cliff at different positions → K matches cliff position exactly
    assert len(adaptive_filter(_items([0.90, 0.88, 0.30, 0.28, 0.25]))) == 2
    assert len(adaptive_filter(_items([0.90, 0.89, 0.88, 0.50, 0.49, 0.48]))) == 3
    assert len(adaptive_filter(_items([0.90, 0.89, 0.88, 0.87, 0.30, 0.28]))) == 4


def test_early_cliff_returns_fewer_than_late_cliff():
    # Early cliff (K=2) must return fewer items than late cliff (K=4)
    early = adaptive_filter(_items([0.90, 0.88, 0.30, 0.28, 0.26, 0.24]))
    late  = adaptive_filter(_items([0.90, 0.89, 0.88, 0.87, 0.30, 0.28]))
    assert len(early) < len(late), (
        f"early cliff should give fewer items ({len(early)}) "
        f"than late cliff ({len(late)})"
    )


def test_respects_k_max():
    scores = [0.9 - i * 0.001 for i in range(ADAPTIVE_K_MAX + 5)]
    result = adaptive_filter(_items(scores))
    assert len(result) <= ADAPTIVE_K_MAX


def test_respects_k_min():
    result = adaptive_filter(_items([0.9, 0.1]))
    assert len(result) >= ADAPTIVE_K_MIN


def test_check_confidence_above():
    assert check_confidence(_items([CONFIDENCE_THRESHOLD + 0.1])) is True


def test_check_confidence_with_items():
    # check_confidence now returns True for any non-empty list,
    # regardless of absolute score value.
    assert check_confidence(_items([CONFIDENCE_THRESHOLD - 0.01])) is True


def test_check_confidence_empty():
    assert check_confidence([]) is False


def test_pipeline_connection():
    """
    Verify adaptive_filter is imported and used by HybridRetriever.
    This confirms the pipeline actually calls dynamic K selection,
    not a fixed top-K.
    """
    import ast
    retriever_src = (
        Path(__file__).parent.parent / "retrieval" / "__init__.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(retriever_src)

    # Check import exists
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "retrieval.adaptive_k"
    ]
    assert imports, "adaptive_filter is not imported in retrieval/__init__.py"

    # Check it is actually called
    assert "adaptive_filter(reranked)" in retriever_src, (
        "adaptive_filter() is imported but never called in HybridRetriever"
    )


# ── Section 2 — Visual demonstration ─────────────────────────────────────

DEMO_DISTRIBUTIONS = [
    {
        "name":   "Clear cliff after position 2",
        "desc":   "Two strong results then a large drop — system stops at 2",
        "scores": [0.92, 0.88, 0.31, 0.28, 0.25, 0.20],
    },
    {
        "name":   "Clear cliff after position 4",
        "desc":   "Four solid results then a cliff — system stops at 4",
        "scores": [0.90, 0.87, 0.84, 0.81, 0.30, 0.28, 0.10],
    },
    {
        "name":   "Late cliff (position 4)",
        "desc":   "Four strong results — big gap at 4 — system stops there",
        "scores": [0.90, 0.89, 0.88, 0.87, 0.30, 0.28, 0.25],
    },
    {
        "name":   "Single strong result",
        "desc":   "One clearly relevant result — system returns exactly 1",
        "scores": [0.95, 0.22, 0.18, 0.15],
    },
    {
        "name":   "Below confidence threshold",
        "desc":   "Top score < 0.30 — out-of-scope, system returns nothing",
        "scores": [0.18, 0.15, 0.12],
    },
]


def run_visual_demo() -> None:
    SEP = "═" * 68

    print(f"\n{SEP}")
    print("  Adaptive K — Elbow Method Visual Demonstration")
    print(f"  (Fixed top-K would always return TOP_K_FINAL={6} items)")
    print(SEP)

    for dist in DEMO_DISTRIBUTIONS:
        items  = _items(dist["scores"])
        result = adaptive_filter(items)
        k_chosen = len(result)

        print(f"\n  Scenario: {dist['name']}")
        print(f"  {dist['desc']}")
        print()

        for i, item in enumerate(items):
            score = item["score"]
            bar   = "█" * int(score * 20)
            chosen = "← INCLUDED" if i < k_chosen else "  excluded"
            arrow  = " ┐" if i == k_chosen - 1 and k_chosen < len(items) else "  "
            print(f"    [{i+1}] {score:.2f}  {bar:<20} {chosen}{arrow}")
            if i == k_chosen - 1 and k_chosen < len(items):
                gap = items[i]["score"] - items[i+1]["score"]
                print(f"         {'─'*20} ← CLIFF (gap={gap:.2f})")

        if k_chosen == 0:
            print(f"\n  → K = 0  (out-of-scope: top score {dist['scores'][0]:.2f} < {CONFIDENCE_THRESHOLD})")
        else:
            print(f"\n  → K = {k_chosen}  (dynamic)  vs  fixed top-K would give K = {min(6, len(items))}")

    print(f"\n{SEP}")
    print("  Conclusion: the system returns between K_MIN and K_MAX items")
    print("  based on where the relevance cliff appears — never a fixed number.")
    print(SEP)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    UNIT_TESTS = [
        test_empty_input_returns_empty,
        test_below_threshold_no_longer_rejects,
        test_single_item_above_threshold,
        test_all_equal_scores_returns_k_min,
        test_clear_cliff_at_position_2,
        test_clear_cliff_at_position_3,
        test_cliff_position_determines_k,
        test_early_cliff_returns_fewer_than_late_cliff,
        test_respects_k_max,
        test_respects_k_min,
        test_check_confidence_above,
        test_check_confidence_with_items,
        test_check_confidence_empty,
        test_pipeline_connection,
    ]

    SEP = "═" * 68
    print(f"\n{SEP}")
    print("  Adaptive K — Unit Tests")
    print(SEP)

    passed = failed = 0
    for t in UNIT_TESTS:
        try:
            t()
            print(f"  ✅  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌  {t.__name__}  —  {e}")
            failed += 1

    print(f"\n  {passed}/{passed + failed} passed")

    run_visual_demo()
