"""
evaluation/eval_intent.py
Intent classifier evaluation — 40-case test suite.

Metrics
-------
- Overall accuracy
- Rules-layer resolution rate (zero API cost)
- Accuracy per difficulty level  (easy / medium / hard)
- Detailed error analysis

Usage
-----
    python evaluation/eval_intent.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.classifier import classify, TEST_CASES

RESULTS_FILE = Path(__file__).parent / "results" / "intent_eval.json"


def main() -> None:
    SEP = "═" * 72

    print(f"\n{SEP}")
    print("  Intent Classifier Evaluation — 40 cases")
    print(SEP)
    print(f"  {'Query':46} {'Expected':26} {'Got':26} Layer")
    print("  " + "─" * 70)

    total     = len(TEST_CASES)
    correct   = 0
    errors:   list[dict] = []
    by_diff:  dict[str, list[int]] = {}
    rules_resolved = 0
    llm_resolved   = 0

    t0 = time.perf_counter()

    for query, expected, _, difficulty in TEST_CASES:
        result = classify(query)
        actual = result["intents"]
        layer  = result["layer"]
        ok     = set(actual) == set(expected)

        if ok:
            correct += 1
        else:
            errors.append({
                "query":      query,
                "expected":   expected,
                "actual":     actual,
                "difficulty": difficulty,
            })

        if layer == "rules":
            rules_resolved += 1
        else:
            llm_resolved += 1

        by_diff.setdefault(difficulty, [0, 0])
        by_diff[difficulty][0] += int(ok)
        by_diff[difficulty][1] += 1

        q_disp = query[:44] + "…" if len(query) > 44 else query
        icon   = "✅" if ok else "❌"
        print(
            f"  {icon} {q_disp:46} "
            f"{str(expected):26} "
            f"{str(actual):26} "
            f"{layer}"
        )

    elapsed = round((time.perf_counter() - t0) * 1000, 0)
    acc     = round(correct / total * 100, 1)

    print(f"\n{SEP}")
    print(f"  Overall Accuracy : {correct}/{total} = {acc}%")
    print(f"  Rules layer      : {rules_resolved} queries resolved (zero API cost)")
    print(f"  LLM layer        : {llm_resolved} queries resolved")
    print(f"  API cost savings : {round(rules_resolved / total * 100, 0):.0f}% of queries free")
    print(f"  Elapsed          : {elapsed:.0f} ms")
    print()

    for diff in ("easy", "medium", "hard"):
        ok_n, tot = by_diff.get(diff, [0, 0])
        pct = round(ok_n / tot * 100, 1) if tot else 0.0
        bar = "█" * int(pct // 10) + "░" * (10 - int(pct // 10))
        print(f"  {diff:8} : {ok_n}/{tot} = {pct:5.1f}%  {bar}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    [{e['difficulty']}] {e['query'][:52]}")
            print(f"      expected: {e['expected']}  |  got: {e['actual']}")

    output = {
        "total":        total,
        "correct":      correct,
        "accuracy":     acc,
        "elapsed_ms":   elapsed,
        "rules_resolved": rules_resolved,
        "llm_resolved":   llm_resolved,
        "by_difficulty": {
            d: {"correct": v[0], "total": v[1],
                "pct": round(v[0] / v[1] * 100, 1) if v[1] else 0}
            for d, v in by_diff.items()
        },
        "errors": errors,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
