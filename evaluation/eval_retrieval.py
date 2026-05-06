"""
evaluation/eval_retrieval.py
Retrieval evaluation — Recall@K, ablation study, out-of-scope rejection.

Test set
--------
40 queries × 8 legal topics (custody, divorce, maintenance, marriage,
lineage, inheritance, procedure) + 5 out-of-scope queries.
Gold standard: multi-label (any acceptable article key counts as a hit).

Usage
-----
    python evaluation/eval_retrieval.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval       import HybridRetriever
from retrieval.rrf   import rrf_merge

RESULTS_FILE = Path(__file__).parent / "results" / "retrieval_eval.json"
K_VALUES     = [1, 3, 6, 10]

TEST_CASES: list[tuple] = [
    # custody
    ("ما هي شروط الحضانة في القانون السوري؟",         ["laws_137", "laws_138", "laws_139"], "custody",     "easy"),
    ("ما هي شروط الحضانة وأسباب سقوطها؟",             ["laws_137", "laws_146"],             "custody",     "easy"),
    ("متى تسقط حضانة الأم؟",                          ["laws_146", "laws_147", "laws_148"], "custody",     "medium"),
    ("ما حق الأب في رؤية أطفاله بعد الطلاق؟",         ["laws_154", "laws_155"],             "custody",     "hard"),
    ("هل يحق للأب استرداد حضانة أولاده؟",             ["laws_149", "laws_150", "laws_151"], "custody",     "hard"),
    # divorce
    ("ما هي أنواع الطلاق في القانون السوري؟",          ["laws_85",  "laws_86"],              "divorce",     "easy"),
    ("ما شروط الطلاق الخلعي؟",                         ["laws_95",  "laws_96"],              "divorce",     "easy"),
    ("متى يحق للزوجة طلب التفريق للضرر؟",             ["laws_105", "laws_106"],             "divorce",     "medium"),
    ("ما الفرق بين الطلاق الرجعي والبائن؟",            ["laws_88",  "laws_89",  "laws_90"],  "divorce",     "hard"),
    ("هل يمكن الطلاق بالتراضي وما إجراءاته؟",          ["laws_91",  "laws_92",  "laws_93"],  "divorce",     "hard"),
    # maintenance
    ("ما حق الزوجة في النفقة بعد الطلاق؟",             ["laws_72",  "laws_73"],              "maintenance", "easy"),
    ("متى تسقط نفقة الزوجة؟",                          ["laws_74",  "laws_75"],              "maintenance", "easy"),
    ("ما نفقة الأطفال وكيف تحدد؟",                     ["laws_76",  "laws_77"],              "maintenance", "medium"),
    ("هل تستحق الزوجة الناشز نفقة؟",                   ["laws_74",  "laws_75",  "laws_78"],  "maintenance", "hard"),
    ("ما مدة نفقة العدة وكيف تحسب؟",                   ["laws_72",  "laws_73",  "laws_74"],  "maintenance", "hard"),
    # marriage
    ("ما شروط صحة عقد الزواج في القانون السوري؟",       ["laws_5",   "laws_6",   "laws_7"],   "marriage",    "easy"),
    ("ما حكم الزواج بدون ولي؟",                         ["laws_14",  "laws_15"],              "marriage",    "easy"),
    ("ما موانع الزواج في القانون السوري؟",               ["laws_25",  "laws_26",  "laws_27"],  "marriage",    "medium"),
    ("ما هي شروط الكفاءة في الزواج؟",                   ["laws_19",  "laws_20"],              "marriage",    "hard"),
    ("هل يصح زواج الأجنبي من سورية؟",                  ["laws_28",  "laws_29"],              "marriage",    "hard"),
    # lineage
    ("كيف يثبت النسب في القانون السوري؟",               ["laws_120", "laws_121"],             "lineage",     "easy"),
    ("ما شروط إثبات النسب بالفراش؟",                    ["laws_120", "laws_122"],             "lineage",     "easy"),
    ("هل يمكن نفي النسب وكيف؟",                         ["laws_125", "laws_126"],             "lineage",     "medium"),
    ("ما حكم الإقرار بالنسب؟",                          ["laws_123", "laws_124"],             "lineage",     "hard"),
    ("ما أثر الزواج الفاسد على النسب؟",                 ["laws_121", "laws_122"],             "lineage",     "hard"),
    # inheritance
    ("ما هي الوصية وشروطها؟",                           ["laws_200", "laws_201"],             "inheritance", "easy"),
    ("هل يرث غير المسلم من المسلم؟",                    ["laws_205", "laws_206"],             "inheritance", "easy"),
    ("ما قواعد الإرث بحسب القانون السوري؟",             ["laws_200", "laws_201", "laws_202"], "inheritance", "medium"),
    ("متى تبطل الوصية؟",                                ["laws_204", "laws_205"],             "inheritance", "hard"),
    ("ما حصة البنت في الميراث؟",                        ["laws_201", "laws_202"],             "inheritance", "hard"),
    # procedure
    ("ما إجراءات تبليغ المدعى عليه؟",                   ["osoul_10", "osoul_11", "osoul_96"], "procedure",   "easy"),
    ("كيف ترفع دعوى أمام محكمة الأحوال الشخصية؟",       ["osoul_96", "osoul_97"],             "procedure",   "easy"),
    ("ما مدة الاستئناف على حكم محكمة الأحوال؟",          ["osoul_150","osoul_151"],            "procedure",   "medium"),
    ("ما شروط قبول الدعوى أمام المحاكم السورية؟",        ["osoul_3",  "osoul_4",  "osoul_5"],  "procedure",   "hard"),
    ("ما حالات رد القضاة عن نظر الدعوى؟",               ["osoul_50", "osoul_51"],             "procedure",   "hard"),
    # out-of-scope
    ("ما هو قانون العقوبات السوري؟",                    [],                                   "out_of_scope","easy"),
    ("شرح نظرية فيثاغورث",                              [],                                   "out_of_scope","easy"),
    ("ما هو سعر صرف الدولار اليوم؟",                   [],                                   "out_of_scope","easy"),
    ("ما هي قوانين العمل في سوريا؟",                    [],                                   "out_of_scope","medium"),
    ("ما هي شروط الترخيص التجاري؟",                     [],                                   "out_of_scope","medium"),
]


# ── Helpers ───────────────────────────────────────────────────────────────

def recall_at_k(results: list[dict], gold_keys: list[str], k: int) -> bool:
    """Return True if any gold key appears in the top-k results."""
    if not gold_keys:
        return False
    return bool({r["key"] for r in results[:k]} & set(gold_keys))


def _rrf_graph(retriever: HybridRetriever, query: str) -> list[dict]:
    """Run full Hybrid RAG pipeline up to graph expansion (before rerank)."""
    fused = rrf_merge(
        retriever._dense.search(query, "legal_laws"),
        retriever._dense.search(query, "legal_osoul"),
        retriever._bm25.search(query,  "legal_laws"),
        retriever._bm25.search(query,  "legal_osoul"),
    )
    if retriever._graph:
        expanded = retriever._graph.expand(fused, depth=1)
        return [i for i in expanded if i["payload"].get("source") != "fonon"]
    return fused


# ── Evaluation routines ───────────────────────────────────────────────────

def run_ablation(retriever: HybridRetriever) -> dict:
    """Compare four retrieval configurations on Recall@K."""
    print("\n" + "═" * 65)
    print("  Ablation Study — Four Retrieval Configurations")
    print("═" * 65)

    legal = [(q, g) for q, g, t, _ in TEST_CASES if g]

    methods = {
        "Dense only": lambda q: (
            retriever._dense.search(q, "legal_laws") +
            retriever._dense.search(q, "legal_osoul")
        ),
        "BM25 only": lambda q: (
            retriever._bm25.search(q, "legal_laws") +
            retriever._bm25.search(q, "legal_osoul")
        ),
        "Dense+BM25+RRF": lambda q: rrf_merge(
            retriever._dense.search(q, "legal_laws"),
            retriever._dense.search(q, "legal_osoul"),
            retriever._bm25.search(q,  "legal_laws"),
            retriever._bm25.search(q,  "legal_osoul"),
        ),
        "RRF+Graph": lambda q: _rrf_graph(retriever, q),
    }

    ablation: dict = {}
    for name, fn in methods.items():
        recalls = {k: [] for k in K_VALUES}
        t0 = time.perf_counter()
        for query, gold in legal:
            res = fn(query)
            for k in K_VALUES:
                recalls[k].append(recall_at_k(res, gold, k))
        avg_ms = round((time.perf_counter() - t0) * 1000 / len(legal), 1)
        ablation[name] = {
            f"Recall@{k}": round(sum(v) / len(v) * 100, 1)
            for k, v in recalls.items()
        }
        ablation[name]["avg_ms"] = avg_ms

    print(f"\n  {'Method':<22} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'R@10':>7} {'ms':>6}")
    print("  " + "─" * 60)
    for name, res in ablation.items():
        print(
            f"  {name:<22} "
            f"{res['Recall@1']:>5.1f}%  "
            f"{res['Recall@3']:>5.1f}%  "
            f"{res['Recall@6']:>5.1f}%  "
            f"{res['Recall@10']:>5.1f}%  "
            f"{res['avg_ms']:>5.1f}"
        )
    return ablation


def run_per_topic(retriever: HybridRetriever) -> dict:
    """Recall@6 breakdown per legal topic using the full pipeline."""
    print("\n" + "═" * 65)
    print("  Per-Topic Recall@6 — Full Pipeline (RRF + Graph)")
    print("═" * 65)

    topics: dict[str, dict] = {}
    for query, gold, topic, _ in TEST_CASES:
        if topic == "out_of_scope":
            continue
        res = _rrf_graph(retriever, query)
        hit = recall_at_k(res, gold, 6)
        topics.setdefault(topic, {"hits": 0, "total": 0})
        topics[topic]["hits"]  += int(hit)
        topics[topic]["total"] += 1

    for topic, d in sorted(topics.items()):
        pct = d["hits"] / d["total"] * 100
        bar = "█" * int(pct // 10) + "░" * (10 - int(pct // 10))
        print(f"  {topic:15} {d['hits']}/{d['total']} = {pct:5.1f}%  {bar}")

    return topics


def run_out_of_scope(retriever: HybridRetriever) -> dict:
    """Verify that out-of-scope queries are rejected (not answered)."""
    print("\n" + "═" * 65)
    print("  Out-of-Scope Rejection Test")
    print("═" * 65)

    oos_cases = [(q, g, t, d) for q, g, t, d in TEST_CASES if t == "out_of_scope"]
    rejected  = 0

    for query, *_ in oos_cases:
        result = retriever.answer_legal_question(query)
        ok     = "error" in result and result["error"] == "out_of_scope"
        if ok:
            rejected += 1
        print(f"  {'✅' if ok else '❌'}  {query[:60]}")

    rate = rejected / len(oos_cases) * 100
    print(f"\n  Rejection rate: {rejected}/{len(oos_cases)} = {rate:.1f}%")

    return {"rejected": rejected, "total": len(oos_cases), "rate": rate}


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 65)
    print("  Retrieval Evaluation — 40 queries, 8 topics, multi-label gold")
    print("═" * 65)

    retriever = HybridRetriever()
    ablation  = run_ablation(retriever)
    topics    = run_per_topic(retriever)
    oos       = run_out_of_scope(retriever)

    best = ablation.get("RRF+Graph", {})
    print("\n  ── Summary (RRF+Graph) ──")
    for k in K_VALUES:
        print(f"  Recall@{k:<3} : {best.get(f'Recall@{k}', 0):.1f}%")
    print(f"  Latency   : {best.get('avg_ms', 0):.1f} ms/query")
    print(f"  OOS Reject: {oos['rate']:.1f}%")

    output = {
        "test_size":    len(TEST_CASES),
        "ablation":     ablation,
        "per_topic":    {k: {"hits": v["hits"], "total": v["total"]}
                         for k, v in topics.items()},
        "out_of_scope": oos,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
