"""
evaluation/eval_llm_judge.py
LLM-as-a-Judge evaluation — response quality on 20 legal questions.

Judge model : llama-3.1-70b-versatile (مختلف عن نموذج التوليد لتجنب self-bias)
Criteria    : Accuracy / Completeness / No-Hallucination / Clarity  (0-10 each)

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
from config   import GROQ_API_KEY, MODEL_JUDGE

RESULTS_FILE = Path(__file__).parent / "results" / "llm_judge_eval.json"

_judge = Groq(api_key=GROQ_API_KEY)

# 20 أسئلة موزعة على كل المواضيع — كافية إحصائياً (كانت 8 فقط)
LEGAL_QUESTIONS: list[str] = [
    # حضانة (3)
    "ما هي شروط الحضانة وأسباب سقوطها في القانون السوري؟",
    "متى تسقط حضانة الأم وكيف يستردها الأب؟",
    "ما حق الأب في رؤية أطفاله بعد الطلاق؟",
    # طلاق (3)
    "ما شروط الطلاق الخلعي؟",
    "ما الفرق بين الطلاق الرجعي والبائن؟",
    "متى يحق للزوجة طلب التفريق للضرر؟",
    # نفقة (3)
    "متى تسقط نفقة الزوجة؟",
    "ما نفقة الأطفال وكيف تحدد؟",
    "ما حق الزوجة في السكن بعد الطلاق؟",
    # زواج (3)
    "ما شروط صحة عقد الزواج في القانون السوري؟",
    "ما موانع الزواج في القانون السوري؟",
    "ما حكم الزواج بدون ولي؟",
    # نسب (2)
    "كيف يثبت النسب أمام المحكمة؟",
    "هل يمكن نفي النسب وكيف؟",
    # إجراءات (3)
    "ما إجراءات تبليغ المدعى عليه في دعوى الأحوال الشخصية؟",
    "كيف ترفع دعوى أمام محكمة الأحوال الشخصية؟",
    "ما مدة الاستئناف على أحكام محكمة الأحوال الشخصية؟",
    # ميراث (2)
    "ما هي الوصية وشروطها في القانون السوري؟",
    "هل يرث غير المسلم من المسلم؟",
    # صعب — متعدد المواضيع (1)
    "ما حقوق الزوجة المالية عند الطلاق من نفقة ومهر وسكن؟",
]

OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "ما هو قانون العقوبات المصري؟",
    "شرح نظرية فيثاغورث",
    "ما هو سعر صرف الدولار اليوم؟",
]

_JUDGE_PROMPT = """\
أنت محكّم محايد تقيّم جودة نظام ذكاء اصطناعي قانوني سوري.

السؤال المطروح:
{question}

إجابة النظام:
{answer}

السياق المتاح للنظام (المواد القانونية المسترجعة):
{context}

قيّم الإجابة على أربعة معايير (0-10 لكل معيار):
1. accuracy          — هل المعلومات القانونية صحيحة ومستندة للسياق؟
2. completeness      — هل غطّت الإجابة الجوانب الرئيسية للسؤال؟
3. no_hallucination  — هل تجنّب النظام اختراع معلومات غير موجودة في السياق؟ (10 = لا هلوسة)
4. clarity           — هل اللغة واضحة ومناسبة للمحامي؟

أرجع فقط JSON صحيح — بدون أي كلام إضافي:
{{"accuracy": X, "completeness": X, "no_hallucination": X, "clarity": X, "comment": "جملة واحدة مختصرة"}}\
"""

_REFUSAL_MARKERS = ["تخصصي", "نطاق", "أصول المحاكمات", "أحوال الشخصية"]


def _judge_response(question: str, answer: str, context: str) -> dict:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        answer=answer[:2500],    
        context=context[:2000],  
    )
    try:
        resp = _judge.chat.completions.create(
            model=MODEL_JUDGE,   
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250,
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


def run_legal_eval() -> list[dict]:
    SEP = "═" * 65
    print(f"\n{SEP}")
    print(f"  LLM-as-a-Judge — {len(LEGAL_QUESTIONS)} Legal Questions")
    print(f"  Judge model: {MODEL_JUDGE}")
    print(SEP)

    scores: list[dict] = []

    for i, question in enumerate(LEGAL_QUESTIONS, 1):
        print(f"  [{i}/{len(LEGAL_QUESTIONS)}] {question[:55]}...")
        try:
            result   = ask(query=question, history=[], last_template=None)
            answer   = result.get("message", "")
            articles = result.get("results", [{}])[0].get("articles", [])

            
            context_parts = []
            for a in articles[:5]:
                text = a.get("text", "")[:300]
                context_parts.append(
                    f"[{a.get('article', '')} — {a.get('law', '')}]\n{text}"
                )
            context = "\n---\n".join(context_parts)

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
            ]) / 4, 1,
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

        time.sleep(1.2)   # rate limit

    return scores


def run_oos_eval() -> list[dict]:
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


def main() -> None:
    scores  = run_legal_eval()
    oos_res = run_oos_eval()

    if not scores:
        print("\n  No scores collected — check API keys.")
        return

    criteria = ("accuracy", "completeness", "no_hallucination", "clarity")
    labels   = {
        "accuracy":         "Legal Accuracy",
        "completeness":     "Completeness",
        "no_hallucination": "No Hallucination",
        "clarity":          "Clarity",
    }

    SEP = "═" * 65
    print(f"\n{SEP}")
    print(f"  Summary ({len(scores)} questions, judge: {MODEL_JUDGE})")
    print(SEP)

    summary: dict[str, float] = {}
    for crit in criteria:
        avg_c = round(sum(s.get(crit, 0) for s in scores) / len(scores), 1)
        summary[crit] = avg_c
        bar = "█" * int(avg_c) + "░" * (10 - int(avg_c))
        print(f"  {labels[crit]:22} : {avg_c:4.1f}/10  {bar}")

    overall = round(sum(s["avg"] for s in scores) / len(scores), 1)
    refused = sum(1 for r in oos_res if r["refused"])
    print(f"  {'Overall Average':22} : {overall:4.1f}/10")
    print(f"  {'OOS Refusal Rate':22} : {refused}/{len(oos_res)}")

    output = {
        "total":        len(scores),
        "judge_model":  MODEL_JUDGE,
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
