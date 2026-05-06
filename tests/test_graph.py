"""
tests/test_graph.py
Unit tests for GraphRetriever.expand() — no external APIs required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.graph.retriever import GraphRetriever, make_key
from config import GRAPH_DECAY


# ── make_key ──────────────────────────────────────────────────────────────

def test_make_key_laws():
    assert make_key({"source": "laws",  "article_num": 137}) == "laws_137"


def test_make_key_osoul():
    assert make_key({"source": "osoul", "article_num": 32})  == "osoul_32"


def test_make_key_fonon():
    assert make_key({"source": "fonon", "template_id": 5})   == "fonon_5"


def test_make_key_missing_source():
    assert make_key({"article_num": 10}) == "_10"


# ── GraphRetriever.expand() ───────────────────────────────────────────────

def _make_retriever(graph: dict, keys: list[str]) -> GraphRetriever:
    payload_index = {k: {"article_num": i, "source": "laws"}
                     for i, k in enumerate(keys)}
    return GraphRetriever(graph, payload_index)


def test_seed_items_always_returned():
    gr    = _make_retriever({"laws_1": ["laws_2"]}, ["laws_1", "laws_2"])
    seeds = [{"key": "laws_1", "payload": {}, "score": 0.9, "source": "dense"}]
    result = gr.expand(seeds, depth=1)
    assert any(r["key"] == "laws_1" for r in result)


def test_neighbour_appended():
    gr    = _make_retriever({"laws_1": ["laws_2"]}, ["laws_1", "laws_2"])
    seeds = [{"key": "laws_1", "payload": {}, "score": 0.8, "source": "dense"}]
    result = gr.expand(seeds, depth=1)
    keys   = [r["key"] for r in result]
    assert "laws_2" in keys


def test_neighbour_score_uses_decay():
    gr    = _make_retriever({"laws_1": ["laws_2"]}, ["laws_1", "laws_2"])
    seed_score = 0.8
    seeds = [{"key": "laws_1", "payload": {}, "score": seed_score, "source": "dense"}]
    result = gr.expand(seeds, depth=1)
    nb     = next(r for r in result if r["key"] == "laws_2")
    assert abs(nb["score"] - round(seed_score * GRAPH_DECAY, 4)) < 1e-6


def test_missing_neighbour_payload_skipped():
    gr = _make_retriever({"laws_1": ["laws_999"]}, ["laws_1"])
    seeds  = [{"key": "laws_1", "payload": {}, "score": 0.9, "source": "dense"}]
    result = gr.expand(seeds, depth=1)
    assert not any(r["key"] == "laws_999" for r in result)


def test_no_duplicates_in_result():
    gr    = _make_retriever(
        {"laws_1": ["laws_2"], "laws_2": ["laws_1"]},
        ["laws_1", "laws_2"],
    )
    seeds = [
        {"key": "laws_1", "payload": {}, "score": 0.9, "source": "dense"},
        {"key": "laws_2", "payload": {}, "score": 0.7, "source": "dense"},
    ]
    result = gr.expand(seeds, depth=1)
    keys   = [r["key"] for r in result]
    assert len(keys) == len(set(keys))


def test_empty_seeds_returns_empty():
    gr = _make_retriever({}, [])
    assert gr.expand([], depth=1) == []


def test_depth_zero_returns_seeds_only():
    gr    = _make_retriever({"laws_1": ["laws_2"]}, ["laws_1", "laws_2"])
    seeds = [{"key": "laws_1", "payload": {}, "score": 0.9, "source": "dense"}]
    result = gr.expand(seeds, depth=0)
    assert [r["key"] for r in result] == ["laws_1"]


if __name__ == "__main__":
    tests = [
        test_make_key_laws,
        test_make_key_osoul,
        test_make_key_fonon,
        test_make_key_missing_source,
        test_seed_items_always_returned,
        test_neighbour_appended,
        test_neighbour_score_uses_decay,
        test_missing_neighbour_payload_skipped,
        test_no_duplicates_in_result,
        test_empty_seeds_returns_empty,
        test_depth_zero_returns_seeds_only,
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
