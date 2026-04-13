#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlp/classifier.py
مصنف النوايا — Waterfall طبقتين

الطبقة 1: Rules (0ms، 0 تكلفة)
  - فارغ / تحية / شكر         → CHAT
  - "المادة X" / "م/X"         → LEGAL_Q
  - صياغة + مرفقات معاً        → [TEMPLATE, ATTACHMENT]

الطبقة 2: Groq llama-3.1-8b-instant
  - كل ما لم تحله الطبقة الأولى

الدالة المُصدَّرة:
    from nlp.classifier import classify
    result = classify("بدي صياغة دعوى طلاق")
    # {"intents": ["TEMPLATE"], "layer": "llm"}
"""

import os
import re
import json
from groq import Groq

VALID_INTENTS = {"TEMPLATE", "ATTACHMENT", "LEGAL_Q", "CHAT"}
from config import MODEL_FAST, GROQ_API_KEY
MODEL = MODEL_FAST

# ── Groq client (داخل دالة لا على مستوى module) ──────────────────────────
def _get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


# ══════════════════════════════════════════════════════════════════════════
# الطبقة 1 — Rules
# ══════════════════════════════════════════════════════════════════════════

# كلمات التحية والدردشة الشائعة في اللهجة السورية والفصحى
_CHAT_TOKENS = {
    "مرحبا", "مرحبتين", "هلا", "هلو", "السلام", "صباح", "مساء",
    "شكرا", "شكراً", "تسلم", "يسلمو", "ممنون", "معك",
    "كيفك", "كيف", "من انت", "من أنت", "عرفني", "اهلا", "أهلاً",
    "وداعا", "باي", "الله", "يعطيك", "أنت", "انت",
}

# regex لرقم المادة الصريح
_ARTICLE_RE = re.compile(
    r'(?:ال|ل|ب)?مادة\s+\d+|م\s*[./]\s*\d+|نص\s+المادة|أحكام\s+المادة',
    re.UNICODE
)

# كلمات تدل على طلب الصياغة
_TEMPLATE_TOKENS = {
    "صياغة", "اكتب", "اكتبلي", "اكتب لي", "حرر", "حررلي",
    "دعوى", "عريضة", "استدعاء", "طلب",
    "بدي", "أريد", "اريد",
}

# كلمات تدل على طلب المرفقات
_ATTACHMENT_TOKENS = {
    "أوراق", "اوراق", "مرفقات", "وثائق", "مستندات",
    "لازم أجيب", "لازم اجيب", "شو لازم", "ايش لازم",
    "ما المطلوب", "ماذا أحضر", "ماذا احضر",
}

# أفعال الكتابة الصريحة — شرط لـ Rule 4 (TEMPLATE+ATTACHMENT)
_WRITE_TOKENS = {
    "اكتب", "اكتبلي", "صياغة", "حرر", "حررلي", "بدي صياغة",
}


def _rules_layer(query: str) -> list | None:
    """
    يُطبّق القواعد البسيطة.
    يُرجع list إذا تأكد، أو None للمرور للطبقة الثانية.
    """
    q       = query.strip()
    q_lower = q.lower()
    tokens  = set(q_lower.replace("؟","").replace("!","").split())

    # ── حالة 1: فارغ ────────────────────────────────────────────
    if not q:
        return ["CHAT"]

    # ── حالة 2: تحية / دردشة واضحة ─────────────────────────────
    # إذا الجملة كلها من كلمات التحية (أقل من 5 توكنات مفيدة)
    if len(tokens) <= 4 and tokens & _CHAT_TOKENS:
        return ["CHAT"]

    # ── حالة 3: رقم مادة صريح بدون طلب صياغة ──────────────────
    if _ARTICLE_RE.search(q):
        # تأكد أنه ليس "اكتب لي المادة 137 في دعوى"
        if not (tokens & _TEMPLATE_TOKENS):
            return ["LEGAL_Q"]

    # ── حالة 4: صياغة + مرفقات معاً — يشترط فعل كتابة صريح ──────
    # "اكتب دعوى + شو الوثائق" → TEMPLATE+ATTACHMENT ✓
    # "ما مستندات دعوى النسب"  → لا فعل كتابة → LLM     ✓
    has_write      = bool(tokens & _WRITE_TOKENS)
    has_attachment = bool(tokens & _ATTACHMENT_TOKENS)
    if has_write and has_attachment:
        return ["TEMPLATE", "ATTACHMENT"]

    # غامض → الطبقة الثانية
    return None


# ══════════════════════════════════════════════════════════════════════════
# الطبقة 2 — Groq LLM
# ══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """أنت مصنف نوايا لمساعد قانوني مخصص للمحامين السوريين.

مهمتك الوحيدة: تحديد نوع طلب المحامي من بين هذه المسارات:

TEMPLATE   = المحامي يطلب منك أنت أن تكتب له صياغة دعوى الآن
             يجب أن يكون فيها فعل طلب صريح: "اكتب" / "صِغ" / "حرر" / "بدي صياغة" / "أريد دعوى"
             أمثلة: "بدي صياغة دعوى طلاق" / "اكتب لي دعوى حضانة" / "حرر عريضة نفقة"
             ⚠ مجرد ذكر كلمة دعوى في سؤال لا يعني TEMPLATE
             ⚠ "شو الأوراق لدعوى الطلاق" → ATTACHMENT فقط وليس TEMPLATE

ATTACHMENT = يسأل عن الوثائق أو المرفقات المطلوبة فقط، لا يطلب كتابة شيء
             أمثلة: "شو الأوراق المطلوبة؟" / "ما هي مستندات دعوى النسب؟" / "ما هي مرفقات دعوى الطلاق؟"

LEGAL_Q    = سؤال قانوني أو يبحث عن حكم أو فرق أو شرط قانوني
             أمثلة: "ما هي شروط الحضانة؟" / "ما الفرق بين الطلاق الرجعي والبائن؟" / "متى تسقط النفقة؟"
             ⚠ "شو الفرق بين دعوى التفريق والخلع" → LEGAL_Q فقط وليس TEMPLATE

CHAT       = دردشة أو سؤال خارج النطاق القانوني تماماً
             أمثلة: "مرحبا" / "كيفك؟" / "شكراً" / "من أنت؟"

قواعد صارمة:
- TEMPLATE فقط عندما يوجد فعل طلب صريح لكتابة صياغة
- ذكر كلمة "دعوى" وحدها لا يكفي لإضافة TEMPLATE
- إذا كان الطلب يحتوي على أكثر من نية حقيقية → أرجع جميعها
- اللهجة السورية العامية مقبولة تماماً
- إذا لم تعرف → أرجع CHAT

أرجع فقط JSON بهذا الشكل بدون أي كلام إضافي:
{"intents": ["INTENT1"]}"""


def _llm_layer(query: str) -> list:
    """يستدعي Groq للحالات الغامضة"""
    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": query.strip()},
            ],
            temperature=0,
            max_tokens=50,
            timeout=10,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        data    = json.loads(raw)
        intents = data.get("intents", [])
        valid   = [i for i in intents if i in VALID_INTENTS]
        return valid if valid else ["CHAT"]

    except json.JSONDecodeError:
        print(f"  ⚠ JSONDecodeError: {query[:50]}")
        return ["CHAT"]
    except Exception as e:
        print(f"  ⚠ LLM error: {e}")
        return ["CHAT"]


# ══════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية
# ══════════════════════════════════════════════════════════════════════════

def classify(query: str) -> dict:
    """
    يُصنف طلب المحامي ويُرجع dict:
      {
        "intents": ["TEMPLATE"],      ← قائمة النوايا
        "layer":   "rules" | "llm"   ← من أي طبقة جاءت
      }
    """
    result = _rules_layer(query)
    if result is not None:
        return {"intents": result, "layer": "rules"}

    result = _llm_layer(query)
    return {"intents": result, "layer": "llm"}


# ══════════════════════════════════════════════════════════════════════════
# اختبار حقيقي — 20 حالة من بيانات المشروع الفعلية
# ══════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── Rules Layer — تحية/دردشة ──────────────────────────────
    ("مرحبا كيفك",                               ["CHAT"],                   "rules"),
    ("شكراً جزيلاً",                              ["CHAT"],                   "rules"),
    ("من أنت؟",                                   ["CHAT"],                   "rules"),

    # ── Rules Layer — رقم مادة صريح ───────────────────────────
    ("المادة 137",                                ["LEGAL_Q"],                "rules"),
    ("نص المادة 95 من قانون الأحوال",             ["LEGAL_Q"],                "rules"),
    ("م/85",                                      ["LEGAL_Q"],                "rules"),

    # ── Rules Layer — صياغة + مرفقات معاً ─────────────────────
    ("بدي صياغة دعوى نفقة وشو الأوراق المطلوبة", ["TEMPLATE", "ATTACHMENT"], "rules"),
    ("اكتب دعوى حضانة وخبرني شو الوثائق",        ["TEMPLATE", "ATTACHMENT"], "rules"),

    # ── LLM Layer — صياغة فقط ─────────────────────────────────
    ("بدي صياغة دعوى طلاق خلعي",                 ["TEMPLATE"],               "llm"),
    ("حرر لي عريضة دعوى تأييد حضانة",            ["TEMPLATE"],               "llm"),
    ("اكتب لي دعوى إسقاط حضانة",                 ["TEMPLATE"],               "llm"),

    # ── LLM Layer — سؤال قانوني فقط ──────────────────────────
    ("ما هي شروط الحضانة وأسباب سقوطها؟",        ["LEGAL_Q"],                "llm"),
    ("متى تسقط نفقة الزوجة؟",                    ["LEGAL_Q"],                "llm"),
    ("شو حكم الخلع إذا رفض الزوج؟",              ["LEGAL_Q"],                "llm"),
    ("ما الفرق بين الطلاق الرجعي والبائن؟",       ["LEGAL_Q"],                "llm"),

    # ── LLM Layer — مرفقات فقط ────────────────────────────────
    ("شو الأوراق المطلوبة لدعوى الطلاق؟",        ["ATTACHMENT"],             "llm"),
    ("ما هي مستندات دعوى إثبات النسب؟",           ["ATTACHMENT"],             "llm"),

    # ── LLM Layer — متعدد صعب ─────────────────────────────────
    ("اكتب دعوى حضانة وخبرني ما هي شروطها قانونياً", ["TEMPLATE", "LEGAL_Q"], "llm"),
    ("شو الفرق بين دعوى التفريق للضرر والخلع",   ["LEGAL_Q"],                "llm"),

    # ── حالة حقيقية صعبة — كلمة دعوى بدون طلب صياغة ──────────
    ("في حال رفع دعوى نفقة ما هي حقوق الزوجة؟", ["LEGAL_Q"],                "llm"),
]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    SEP = "═" * 65

    print(f"\n{SEP}")
    print("  اختبار Intent Classifier — 20 حالة حقيقية")
    print(SEP)
    print(f"  {'السؤال':42} {'متوقع':25} {'فعلي':25} {'طبقة'}")
    print("  " + "─" * 63)

    passed = failed = rules_count = llm_count = 0

    for query, expected_intents, expected_layer in TEST_CASES:
        result          = classify(query)
        actual_intents  = result["intents"]
        actual_layer    = result["layer"]

        intents_ok = set(actual_intents) == set(expected_intents)
        layer_ok   = actual_layer == expected_layer
        ok         = intents_ok  # الطبقة للمعلومة فقط، لا تؤثر على النجاح

        if ok:
            passed += 1
            icon = "✅"
        else:
            failed += 1
            icon = "❌"

        if actual_layer == "rules":
            rules_count += 1
        else:
            llm_count += 1

        q_display = query[:40] + "…" if len(query) > 40 else query
        e_display = str(expected_intents)
        a_display = str(actual_intents)
        layer_display = f"{actual_layer}" + ("" if layer_ok else f" (متوقع: {expected_layer})")

        print(f"  {icon}  {q_display:42} {e_display:25} {a_display:25} {layer_display}")

    print(f"\n{SEP}")
    print(f"  النتيجة : {passed}/{len(TEST_CASES)} صحيح")
    print(f"  Rules   : {rules_count} طلب بدون API")
    print(f"  LLM     : {llm_count} طلب استدعى Groq")
    print(f"  توفير   : {rules_count/len(TEST_CASES)*100:.0f}% من الطلبات بدون تكلفة")
    print(SEP)
