"""
evaluation/eval_llm_judge.py
LLM-as-a-Judge evaluation — response quality on 8 legal questions.

Judge model : LLaMA-3.3-70b-versatile (independent from the generation model)
Criteria    : Accuracy / Completeness / No-Hallucination / Clarity  (0–10 each)

Also tests out-of-scope refusal: the system must politely decline
non-legal queries rather than hallucinating an answer.

Usage
-----
    python evaluation/eval_llm_judge.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from groq     import Groq
from pipeline import ask
from config   import GROQ_API_KEY, MODEL_LEGAL

RESULTS_FILE = Path(__file__).parent / "results" / "llm_judge_eval.json"
JUDGE_MODEL  = MODEL_LEGAL

_judge = Groq(api_key=GROQ_API_KEY)

LEGAL_QUESTIONS: list[str] = [
    "ما هي شروط الحضانة وأسباب سقوطها في القانون السوري؟",
    "ما إجراءات تبليغ المدعى عليه في دعوى الأحوال الشخصية؟",
    "متى تسقط نفقة الزوجة؟",
    "ما شروط الطلاق الخلعي؟",
    "كيف يثبت النسب أمام المحكمة؟",
    "ما حق الزوجة في السكن بعد الطلاق؟",
    "ما مدة الاستئناف على أحكام محكمة الأحوال الشخصية؟",
    "ما شروط صحة عقد الزواج في القانون السوري؟",
]

OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "ما هو قانون العقوبات السوري؟",
    "شرح نظرية فيثاغورث",
    "ما هو سعر صرف الدولار اليوم؟",
]

_JUDGE_PROMPT = """\
You are a neutral AI evaluator assessing a Syrian legal AI assistant.

Question asked:
{question}

System response:
{answer}

Retrieved context (what the system had available):
{context}

Rate the response on these four criteria (0-10 each):
1. accuracy          — Are all legal facts drawn from the context and correct?
2. completeness      — Does the response cover the main aspects of the question?
3. no_hallucination  — Did the system avoid generating facts NOT in the context? (10 = zero hallucination)
4. clarity           — Is the language appropriate and clear for a lawyer?

Reply ONLY with valid JSON — no extra text:
{{"accuracy": X, "completeness": X, "no_hallucination": X, "clarity": X, "comment": "one short sentence"}}\
"""

# Keywords that indicate a proper scoped refusal
_REFUSAL_MARKERS = ["تخصصي", "نطاق", "أصول المحاكمات", "أحوال الشخصية"]


# ── Judge call ────────────────────────────────────────────────────────────

def _judge_response(question: str, answer: str, context: str) -> dict:
    """Ask the judge model to score a single system response."""
    prompt = _JUDGE_PROMPT.format(
        question=question,
        answer=answer[:800],
        context=context[:600],
    )
    try:
        resp = _judge.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = (
            resp.choices[0].message.content
            .strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        return json.loads(raw)
    except Exception as exc:
        print(f"  [Judge] error: {exc}")
        return {
            "accuracy": 0, "completeness": 0,
            "no_hallucination": 0, "clarity": 0,
            "comment": "error",
        }


# ── Evaluation routines ───────────────────────────────────────────────────

def run_legal_eval() -> list[dict]:
    """Evaluate the system on LEGAL_QUESTIONS using the LLM judge."""
    SEP = "═" * 65
    print(f"\n{SEP}")
    print(f"  LLM-as-a-Judge — Legal Questions ({len(LEGAL_QUESTIONS)})")
    print(SEP)

    scores: list[dict] = []
    n = len(LEGAL_QUESTIONS)

    for i, question in enumerate(LEGAL_QUESTIONS, 1):
        print(f"  [{i}/{n}] {question[:55]}...")
        try:
            result   = ask(query=question, history=[], last_template=None)
            answer   = result.get("message", "")
            articles = result.get("results", [{}])[0].get("articles", [])
            context  = "\n".join(
                f"[{a.get('article', '')}] {a.get('text', '')[:100]}"
                for a in articles[:3]
            )
        except Exception as exc:
            print(f"    [System] error: {exc}")
            continue

        s   = _judge_response(question, answer, context)
        avg = round(
            sum([
                s.get("accuracy", 0),
                s.get("completeness", 0),
                s.get("no_hallucination", 0),
                s.get("clarity", 0),
            ]) / 4,
            1,
        )
        s.update({"avg": avg, "question": question})
        scores.append(s)

        print(
            f"    Acc={s['accuracy']}  Com={s['completeness']}  "
            f"NoHal={s['no_hallucination']}  Clar={s['clarity']}  "
            f"→ {avg}/10"
        )
        if s.get("comment"):
            print(f"    Note: {s['comment']}")

        time.sleep(1.0)   # respect Groq rate limits

    return scores


def run_oos_eval() -> list[dict]:
    """Verify that out-of-scope queries receive a scoped refusal."""
    SEP = "═" * 65
    print(f"\n{SEP}")
    print("  Out-of-Scope Refusal Check")
    print(SEP)

    results: list[dict] = []
    for question in OUT_OF_SCOPE_QUESTIONS:
        result  = ask(query=question, history=[], last_template=None)
        message = result.get("message", "")
        refused = any(marker in message for marker in _REFUSAL_MARKERS)
        icon    = "✅" if refused else "❌"
        print(f"  {icon}  {question[:55]}")
        if not refused:
            print(f"      Response: {message[:80]}…")
        results.append({"question": question, "refused": refused})

    return results


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    scores  = run_legal_eval()
    oos_res = run_oos_eval()

    if not scores:
        print("\n  No scores collected — check API keys and connectivity.")
        return

    criteria = ("accuracy", "completeness", "no_hallucination", "clarity")
    labels   = {
        "accuracy":        "Legal Accuracy",
        "completeness":    "Completeness",
        "no_hallucination":"No Hallucination",
        "clarity":         "Clarity",
    }

    SEP = "═" * 65
    print(f"\n{SEP}")
    print(f"  Summary ({len(scores)} questions evaluated)")
    print(SEP)

    summary: dict[str, float] = {}
    for crit in criteria:
        avg_c = round(sum(s.get(crit, 0) for s in scores) / len(scores), 1)
        summary[crit] = avg_c
        bar = "█" * int(avg_c) + "░" * (10 - int(avg_c))
        print(f"  {labels[crit]:22} : {avg_c:4.1f}/10  {bar}")

    overall  = round(sum(s["avg"] for s in scores) / len(scores), 1)
    refused  = sum(1 for r in oos_res if r["refused"])
    print(f"  {'Overall Average':22} : {overall:4.1f}/10  ({round(overall * 10, 1)}%)")
    print(f"  {'OOS Refusal Rate':22} : {refused}/{len(oos_res)}")

    output = {
        "total":        len(scores),
        "judge_model":  JUDGE_MODEL,
        "scores":       scores,
        "summary":      summary,
        "overall_avg":  overall,
        "out_of_scope": {
            "refused": refused,
            "total":   len(oos_res),
            "rate":    round(refused / len(oos_res) * 100, 1) if oos_res else 0,
        },
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
