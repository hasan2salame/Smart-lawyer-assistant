"""
tests/test_bm25.py
Unit tests for BM25 tokeniser and scoring — no external APIs required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.bm25 import _tokenize, BM25Searcher


# ── Tokeniser tests ───────────────────────────────────────────────────────

def test_removes_tashkeel():
    assert _tokenize("الزَّوْجِيَّة") == ["الزوجيه"]


def test_normalises_hamza():
    tokens = _tokenize("أحكام إثبات آمر")
    # All hamza variants → bare alef
    for t in tokens:
        assert "أ" not in t and "إ" not in t and "آ" not in t


def test_normalises_ta_marbuta():
    tokens = _tokenize("محكمة الأحوال")
    assert any("محكمه" in t for t in tokens)


def test_filters_stop_words():
    tokens = _tokenize("في من على الزواج والطلاق")
    assert "في" not in tokens
    assert "من" not in tokens
    assert "على" not in tokens


def test_filters_short_tokens():
    tokens = _tokenize("في و أ الطلاق")
    assert all(len(t) > 2 for t in tokens)


def test_empty_string():
    assert _tokenize("") == []


def test_returns_list():
    assert isinstance(_tokenize("قانون الأحوال الشخصية"), list)


# ── BM25Searcher index tests ──────────────────────────────────────────────

def _make_searcher(docs: dict[str, str]) -> BM25Searcher:
    """Build a minimal BM25Searcher from a string → text mapping."""
    col = "legal_laws"
    payload_index = {k: {"original_text": v, "source": "laws", "article_num": i}
                     for i, (k, v) in enumerate(docs.items())}
    corpus_keys   = {col: list(docs.keys())}
    return BM25Searcher(payload_index=payload_index, corpus_keys=corpus_keys)


def test_relevant_doc_ranks_first():
    searcher = _make_searcher({
        "laws_1": "شروط الحضانة وانتقالها بين الوالدين",
        "laws_2": "أحكام المهر والصداق في عقد الزواج",
        "laws_3": "إجراءات تبليغ المدعى عليه في الدعوى",
    })
    results = searcher.search("شروط الحضانة", "legal_laws")
    assert results[0]["key"] == "laws_1"


def test_empty_query_returns_empty():
    searcher = _make_searcher({"laws_1": "نص المادة"})
    assert searcher.search("", "legal_laws") == []


def test_unknown_collection_returns_empty():
    searcher = _make_searcher({"laws_1": "نص المادة"})
    assert searcher.search("حضانة", "legal_fonon") == []


def test_scores_descending():
    searcher = _make_searcher({
        "laws_1": "الطلاق الخلعي وشروطه في القانون السوري",
        "laws_2": "نفقة الزوجة بعد الطلاق",
        "laws_3": "أحكام الميراث والوصية",
    })
    results = searcher.search("الطلاق", "legal_laws")
    scores  = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_source_field_is_bm25():
    searcher = _make_searcher({"laws_1": "حضانة الأطفال"})
    results  = searcher.search("حضانة", "legal_laws")
    if results:
        assert results[0]["source"] == "bm25"


if __name__ == "__main__":
    tests = [
        test_removes_tashkeel,
        test_normalises_hamza,
        test_normalises_ta_marbuta,
        test_filters_stop_words,
        test_filters_short_tokens,
        test_empty_string,
        test_returns_list,
        test_relevant_doc_ranks_first,
        test_empty_query_returns_empty,
        test_unknown_collection_returns_empty,
        test_scores_descending,
        test_source_field_is_bm25,
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
