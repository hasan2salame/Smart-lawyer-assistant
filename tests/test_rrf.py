"""
tests/test_rrf.py
Unit tests for rrf_merge() — no external APIs required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.rrf import rrf_merge
from config import RRF_K, TOP_K_RRF


def _items(keys: list[str]) -> list[dict]:
    return [{"key": k, "payload": {"article_num": i}, "score": 1.0}
            for i, k in enumerate(keys)]


def test_single_list_preserves_order():
    result = rrf_merge(_items(["a", "b", "c"]))
    assert [r["key"] for r in result] == ["a", "b", "c"]
    assert result[0]["source"] == "rrf"


def test_disjoint_lists_all_appear():
    result = rrf_merge(_items(["a", "b"]), _items(["c", "d"]))
    assert set(r["key"] for r in result) == {"a", "b", "c", "d"}


def test_shared_key_gets_highest_score():
    l1 = _items(["a", "b", "c"])
    l2 = _items(["a", "d", "e"])
    result = rrf_merge(l1, l2)
    scores = {r["key"]: r["score"] for r in result}
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["d"]


def test_empty_input_returns_empty():
    assert rrf_merge([], []) == []


def test_respects_top_k_limit():
    keys   = [f"doc_{i}" for i in range(TOP_K_RRF + 10)]
    result = rrf_merge(_items(keys))
    assert len(result) <= TOP_K_RRF


def test_score_formula_correctness():
    result = rrf_merge([{"key": "x", "payload": {}, "score": 0.9}])
    expected = 1.0 / (RRF_K + 1)
    assert abs(result[0]["score"] - expected) < 1e-9


def test_item_without_key_is_ignored():
    l1 = [{"key": "a", "payload": {}, "score": 1.0}]
    l2 = [{"payload": {}, "score": 0.9}]   # missing "key"
    result = rrf_merge(l1, l2)
    assert len(result) == 1 and result[0]["key"] == "a"


def test_payload_preserved():
    l1 = [{"key": "a", "payload": {"law_name": "test"}, "score": 1.0}]
    result = rrf_merge(l1)
    assert result[0]["payload"]["law_name"] == "test"


if __name__ == "__main__":
    tests = [
        test_single_list_preserves_order,
        test_disjoint_lists_all_appear,
        test_shared_key_gets_highest_score,
        test_empty_input_returns_empty,
        test_respects_top_k_limit,
        test_score_formula_correctness,
        test_item_without_key_is_ignored,
        test_payload_preserved,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌  {t.__name__}  —  {e}")
            failed += 1
    print(f"\n  {passed}/{passed + failed} passed")
