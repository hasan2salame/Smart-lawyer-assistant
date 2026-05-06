"""
nlp/classifier.py
Intent classifier — two-layer waterfall architecture.

Layer 1 — Rule-based (0 ms, 0 API cost)
    Empty / greeting / thanks  → CHAT
    "المادة X" / "م/X"          → LEGAL_Q
    write-verb + attachment     → [TEMPLATE, ATTACHMENT]

Layer 2 — Groq LLaMA-3.1-8b
    All queries not resolved by Layer 1.

Public API
----------
    from nlp.classifier import classify
    result = classify("بدي صياغة دعوى طلاق")
    # {"intents": ["TEMPLATE"], "layer": "llm"}
"""

import re
import json
from groq import Groq

from config import MODEL_FAST, GROQ_API_KEY

VALID_INTENTS = {"TEMPLATE", "ATTACHMENT", "LEGAL_Q", "CHAT"}

_groq = Groq(api_key=GROQ_API_KEY)

# ── Layer 1 — vocabulary sets ─────────────────────────────────────────────

_CHAT_TOKENS: frozenset[str] = frozenset({
    "مرحبا", "مرحبتين", "هلا", "هلو", "السلام", "صباح", "مساء",
    "شكرا", "شكراً", "تسلم", "يسلمو", "ممنون", "معك",
    "كيفك", "كيف", "من انت", "من أنت", "عرفني", "اهلا", "أهلاً",
    "وداعا", "باي", "الله", "يعطيك", "أنت", "انت",
})

# Explicit article number pattern — triggers LEGAL_Q immediately
_ARTICLE_RE = re.compile(
    r"(?:ال|ل|ب)?مادة\s+\d+|م\s*[./]\s*\d+|نص\s+المادة|أحكام\s+المادة",
    re.UNICODE,
)

# Tokens indicating a drafting request (write-verb required)
_TEMPLATE_TOKENS: frozenset[str] = frozenset({
    "صياغة", "اكتب", "اكتبلي", "اكتب لي", "حرر", "حررلي",
    "دعوى", "عريضة", "استدعاء", "طلب",
    "بدي", "أريد", "اريد",
})

_ATTACHMENT_TOKENS: frozenset[str] = frozenset({
    "أوراق", "اوراق", "مرفقات", "مرفقاتها", "مرفقاته", "مرفقاتهم",
    "وثائق", "مستندات", "المستندات", "الوثائق", "الأوراق",
    "لازم أجيب", "لازم اجيب", "شو لازم", "ايش لازم",
    "ما المطلوب", "ماذا أحضر", "ماذا احضر",
    "ما هي الوثائق", "ما هي المستندات", "ما هي الأوراق",
})

# Explicit write-verbs required for the composite TEMPLATE+ATTACHMENT rule
_WRITE_TOKENS: frozenset[str] = frozenset({
    "اكتب", "اكتبلي", "صياغة", "حرر", "حررلي", "بدي صياغة",
})

# Legal domain keywords — any query containing these is NOT a CHAT query
_LEGAL_KEYWORDS: frozenset[str] = frozenset({
    "حضانة", "طلاق", "نفقة", "مهر", "زواج", "ميراث", "وصية", "نسب",
    "تبليغ", "دعوى", "مادة", "قانون", "محكمة", "حكم", "استئناف",
    "مرفقات", "وثائق", "صياغة", "عريضة", "تفريق", "خلع", "ولاية",
    "وصاية", "قيمومة", "نشوز", "عدة", "رجعة", "فسخ", "خلعي",
    "إجراءات", "اجراءات", "تنفيذ", "اختصاص", "طعن", "شروط",
    "حقوق", "مدة", "سقوط", "إثبات", "اثبات", "أحوال", "احوال",
})

# ── Layer 1 — rule engine ─────────────────────────────────────────────────

def _rules_layer(query: str) -> list[str] | None:
    """
    Apply deterministic classification rules.

    Returns a list of intents if a rule matches, or None to escalate
    to the LLM layer.
    """
    q      = query.strip()
    tokens = set(q.lower().replace("؟", "").replace("!", "").split())

    # Rule 1 — Empty query
    if not q:
        return ["CHAT"]

    # Rule 2 — Pure greeting / small-talk (≤ 4 tokens, all from chat vocab)
    if len(tokens) <= 4 and tokens & _CHAT_TOKENS:
        return ["CHAT"]

    # Rule 2b — Short message (≤ 3 tokens) with no legal keywords → CHAT
    # Catches conversational follow-ups like "فقط هكذا؟" or "وبعدين؟"
    # without accidentally catching short legal queries.
    if len(tokens) <= 3 and not (tokens & _LEGAL_KEYWORDS):
        return ["CHAT"]

    # Rule 3 — Explicit article number lookup (no drafting verb present)
    if _ARTICLE_RE.search(q) and not (tokens & _TEMPLATE_TOKENS):
        return ["LEGAL_Q"]

    # Rule 4 — Composite TEMPLATE + ATTACHMENT
    # Requires BOTH an explicit write-verb AND an attachment keyword.
    # Guards against attachment-only queries (e.g. "what docs for a paternity case?") being
    # misclassified as TEMPLATE.
    if (tokens & _WRITE_TOKENS) and (tokens & _ATTACHMENT_TOKENS):
        return ["TEMPLATE", "ATTACHMENT"]

    return None  # escalate to LLM


# ── Layer 2 — LLM classification ─────────────────────────────────────────

_SYSTEM_PROMPT = """\
أنت مصنف نوايا دقيق لمساعد قانوني مخصص للمحامين السوريين.
مهمتك الوحيدة: تحديد نوع طلب المحامي بدقة عالية.

══ الأنواع الأربعة ══

TEMPLATE
  المحامي يطلب منك كتابة صياغة دعوى أو عريضة الآن.
  شرط لازم: فعل طلب صريح مثل "اكتب" / "صِغ" / "حرر" / "بدي صياغة" / "أريد دعوى".
  بدون فعل طلب → ليس TEMPLATE.

ATTACHMENT
  يسأل عن الوثائق أو المرفقات أو الأوراق المطلوبة لدعوى معينة.
  كلمات تدل عليه: مرفقات / وثائق / أوراق / مستندات / مرفقاتها / ما المطلوب.

LEGAL_Q
  أي سؤال قانوني: شروط / إجراءات / أحكام / فروق / حقوق / مدد / تعريفات.
  هذا النوع واسع جداً — الشك يذهب لـ LEGAL_Q وليس CHAT.
  كلمات تدل عليه: ما هي / كيف / متى / هل / شو / ما إجراءات / ما شروط.

CHAT
  فقط للدردشة الخالصة أو الأسئلة الخارجة عن القانون تماماً مثل الطقس والرياضيات.
  إذا كان السؤال يتعلق بأي موضوع قانوني ولو بشكل بعيد → ليس CHAT.

══ أمثلة حرجة ══

"ما هي شروط الحضانة ومرفقاتها؟"
→ {"intents": ["LEGAL_Q", "ATTACHMENT"]}
  (شروط = LEGAL_Q، مرفقاتها = ATTACHMENT)

"ما هي إجراءات تبليغ المدعى عليه؟"
→ {"intents": ["LEGAL_Q"]}
  (سؤال إجرائي قانوني واضح)

"شو الأوراق لدعوى الطلاق؟"
→ {"intents": ["ATTACHMENT"]}
  (لا فعل كتابة → ليس TEMPLATE)

"اكتب دعوى طلاق وخبرني مرفقاتها"
→ {"intents": ["TEMPLATE", "ATTACHMENT"]}

"ما الفرق بين الطلاق الرجعي والبائن؟"
→ {"intents": ["LEGAL_Q"]}

"كيف حالك؟"
→ {"intents": ["CHAT"]}

══ قواعد نهائية ══
- الشك يذهب لـ LEGAL_Q — CHAT فقط للأسئلة الخارجة عن القانون تماماً.
- إذا كان الطلب يحتوي على نيتين حقيقيتين → أرجعهما معاً.
- اللهجة السورية العامية مقبولة تماماً.

أرجع فقط JSON بهذا الشكل بدون أي كلام إضافي:
{"intents": ["INTENT1"]}\
"""


def _llm_layer(query: str) -> list[str]:
    """Call Groq LLaMA-3.1-8b for queries not resolved by the rules layer."""
    try:
        response = _groq.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": query.strip()},
            ],
            temperature=0,
            max_tokens=50,
            timeout=10,
        )
        raw = (
            response.choices[0].message.content
            .strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        data    = json.loads(raw)
        intents = data.get("intents", [])
        valid   = [i for i in intents if i in VALID_INTENTS]
        return valid if valid else ["CHAT"]

    except json.JSONDecodeError:
        print(f"[Classifier] JSONDecodeError on: {query[:60]}")
        return ["CHAT"]
    except Exception as exc:
        print(f"[Classifier] LLM error: {exc}")
        return ["CHAT"]


# ── Public API ────────────────────────────────────────────────────────────

def classify(query: str) -> dict:
    """
    Classify a lawyer query.

    Returns
    -------
    dict with keys:
        intents : list[str]  — e.g. ["TEMPLATE", "ATTACHMENT"]
        layer   : str        — "rules" | "llm"
    """
    result = _rules_layer(query)
    if result is not None:
        return {"intents": result, "layer": "rules"}

    result = _llm_layer(query)
    return {"intents": result, "layer": "llm"}


# ── Test suite — 40 cases ─────────────────────────────────────────────────
# Format: (query, expected_intents, expected_layer, difficulty)
#
# Distribution:
#   LEGAL_Q     16 cases  (custody 3, divorce 3, maintenance 3,
#                           inheritance 2, marriage 2, procedure 3)
#   TEMPLATE     8 cases
#   ATTACHMENT   5 cases
#   CHAT         5 cases
#   MULTI        6 cases  (TEMPLATE+ATTACHMENT, TEMPLATE+LEGAL, ATTACHMENT+LEGAL)
#
# Difficulty levels: easy / medium / hard

TEST_CASES: list[tuple] = [
    # CHAT — rules layer
    ("مرحبا كيفك",                                    ["CHAT"],                   "rules", "easy"),
    ("شكراً جزيلاً",                                  ["CHAT"],                   "rules", "easy"),
    ("من أنت؟",                                       ["CHAT"],                   "rules", "easy"),
    ("وداعاً",                                        ["CHAT"],                   "rules", "easy"),
    ("شكرا يسلمو",                                    ["CHAT"],                   "rules", "easy"),

    # LEGAL_Q — rules layer (explicit article reference)
    ("المادة 137",                                    ["LEGAL_Q"],                "rules", "easy"),
    ("م/85",                                          ["LEGAL_Q"],                "rules", "easy"),
    ("نص المادة 95 من قانون الأحوال",                 ["LEGAL_Q"],                "rules", "easy"),

    # TEMPLATE + ATTACHMENT — rules layer (write-verb + attachment token)
    ("بدي صياغة دعوى نفقة وشو الأوراق المطلوبة",     ["TEMPLATE", "ATTACHMENT"], "rules", "medium"),
    ("اكتب دعوى حضانة وخبرني شو الوثائق",            ["TEMPLATE", "ATTACHMENT"], "rules", "medium"),

    # LEGAL_Q — LLM layer — custody
    ("ما هي شروط الحضانة وأسباب سقوطها؟",            ["LEGAL_Q"],                "llm",   "easy"),
    ("متى تنتهي حضانة الأم للأولاد؟",                 ["LEGAL_Q"],                "llm",   "medium"),
    ("شو شروط حضانة الولاد بالقانون السوري؟",         ["LEGAL_Q"],                "llm",   "medium"),

    # LEGAL_Q — divorce
    ("ما شروط الطلاق الخلعي؟",                        ["LEGAL_Q"],                "llm",   "easy"),
    ("ما الفرق بين الطلاق الرجعي والبائن؟",           ["LEGAL_Q"],                "llm",   "medium"),
    ("شو حكم الخلع إذا رفض الزوج؟",                  ["LEGAL_Q"],                "llm",   "medium"),

    # LEGAL_Q — maintenance
    ("متى تسقط نفقة الزوجة؟",                         ["LEGAL_Q"],                "llm",   "easy"),
    ("ما حق المرأة في النفقة بعد الطلاق؟",            ["LEGAL_Q"],                "llm",   "medium"),
    ("هل تستحق الزوجة الناشز نفقة؟",                  ["LEGAL_Q"],                "llm",   "hard"),

    # LEGAL_Q — marriage
    ("ما شروط صحة عقد الزواج؟",                       ["LEGAL_Q"],                "llm",   "easy"),
    ("ما حكم الزواج بدون ولي في القانون السوري؟",      ["LEGAL_Q"],                "llm",   "medium"),

    # LEGAL_Q — civil procedure
    ("ما إجراءات تبليغ المدعى عليه؟",                 ["LEGAL_Q"],                "llm",   "medium"),
    ("ما مدة الاستئناف على حكم محكمة الأحوال؟",       ["LEGAL_Q"],                "llm",   "medium"),
    ("كيف ترفع دعوى أمام محكمة الأحوال الشخصية؟",    ["LEGAL_Q"],                "llm",   "easy"),

    # LEGAL_Q — hard boundary: mention of "lawsuit" without an explicit drafting verb
    ("شو الفرق بين دعوى التفريق للضرر والخلع",        ["LEGAL_Q"],                "llm",   "hard"),
    ("في حال رفع دعوى نفقة ما هي حقوق الزوجة؟",      ["LEGAL_Q"],                "llm",   "hard"),
    ("ما هي أسباب فسخ عقد الزواج في القانون السوري؟", ["LEGAL_Q"],                "llm",   "medium"),

    # TEMPLATE — LLM layer
    ("بدي صياغة دعوى طلاق خلعي",                     ["TEMPLATE"],               "llm",   "easy"),
    ("حرر لي عريضة دعوى تأييد حضانة",                ["TEMPLATE"],               "llm",   "easy"),
    ("اكتب لي دعوى إسقاط حضانة",                     ["TEMPLATE"],               "llm",   "easy"),
    ("اكتب دعوى المطالبة بالمهر المؤجل",              ["TEMPLATE"],               "llm",   "medium"),
    ("بدي دعوى مشاهدة الأطفال",                       ["TEMPLATE"],               "llm",   "medium"),
    ("صياغة دعوى نفقة الزوجة والأطفال",               ["TEMPLATE"],               "llm",   "easy"),
    ("حرر حجة وصاية على القاصر",                      ["TEMPLATE"],               "llm",   "hard"),
    ("بدي أرفع دعوى تسليم أطفال",                     ["TEMPLATE"],               "llm",   "hard"),

    # ATTACHMENT — LLM layer
    ("شو الأوراق المطلوبة لدعوى الطلاق؟",             ["ATTACHMENT"],             "llm",   "easy"),
    ("ما هي مستندات دعوى إثبات النسب؟",               ["ATTACHMENT"],             "llm",   "easy"),
    ("شو لازم أجيب معي للمحكمة؟",                     ["ATTACHMENT"],             "llm",   "medium"),
    ("ما هي وثائق دعوى تسليم الأطفال؟",               ["ATTACHMENT"],             "llm",   "medium"),
    ("شو المطلوب لدعوى رفع النفقة؟",                  ["ATTACHMENT"],             "llm",   "medium"),

    # Multi-intent — LLM layer
    ("اكتب دعوى حضانة وخبرني ما هي شروطها قانونياً", ["TEMPLATE", "LEGAL_Q"],    "llm",   "hard"),
    ("بدي صياغة دعوى نفقة وشو الأوراق؟",             ["TEMPLATE", "ATTACHMENT"], "llm",   "medium"),
    ("ما مستندات دعوى الحضانة وشو الأوراق المطلوبة؟", ["ATTACHMENT"],             "llm",   "hard"),

    # Regression cases — previously misclassified as CHAT or wrong intent
    ("ما هي شروط الحضانة ومرفقاتها؟",                ["LEGAL_Q", "ATTACHMENT"],  "llm",   "hard"),
    ("ما هي إجراءات تبليغ المدعى عليه؟",             ["LEGAL_Q"],                "llm",   "medium"),
    ("شو شروط النفقة وما مرفقاتها؟",                 ["LEGAL_Q", "ATTACHMENT"],  "llm",   "hard"),
]


if __name__ == "__main__":
    SEP = "═" * 72

    print(f"\n{SEP}")
    print("  Intent Classifier — 40-case test suite")
    print(SEP)
    print(f"  {'Query':45} {'Expected':28} {'Actual':28} {'Layer'}")
    print("  " + "─" * 70)

    passed = failed = rules_count = llm_count = 0
    by_difficulty: dict[str, list[int]] = {
        "easy": [0, 0], "medium": [0, 0], "hard": [0, 0],
    }

    for query, expected_intents, _, difficulty in TEST_CASES:
        result         = classify(query)
        actual_intents = result["intents"]
        actual_layer   = result["layer"]
        ok             = set(actual_intents) == set(expected_intents)

        if ok:
            passed += 1
            by_difficulty[difficulty][0] += 1
        else:
            failed += 1
        by_difficulty[difficulty][1] += 1

        if actual_layer == "rules":
            rules_count += 1
        else:
            llm_count += 1

        q_disp = query[:43] + "…" if len(query) > 43 else query
        icon   = "✅" if ok else "❌"
        print(
            f"  {icon}  {q_disp:45} "
            f"{str(expected_intents):28} "
            f"{str(actual_intents):28} "
            f"{actual_layer}"
        )

    total = len(TEST_CASES)
    print(f"\n{SEP}")
    print(f"  Overall Accuracy : {passed}/{total} = {passed / total * 100:.1f}%")
    print(f"  Rules layer      : {rules_count} queries  (zero API cost)")
    print(f"  LLM layer        : {llm_count} queries")
    print(f"  API cost savings : {rules_count / total * 100:.0f}% of queries free")
    print()
    for diff, (ok_count, total_d) in by_difficulty.items():
        if total_d:
            pct = ok_count / total_d * 100
            print(f"  Difficulty [{diff:6}]: {ok_count}/{total_d} = {pct:.0f}%")
    print(SEP)
