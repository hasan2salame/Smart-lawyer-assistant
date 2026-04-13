#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlp/llm.py
طبقة الاستجابة الذكية

المعمارية:
  [1] كشف السيناريو (تعبئة / جديد)
  [2] معالجة كل intent بـ handler مستقل (لا خلط في السياق)
  [3] دمج النتائج بترتيب ثابت

الإصلاحات:
  ✓ prompt بسيط لكل handler بدل 7 فروع معقدة
  ✓ context منفصل لكل intent → لا تكرار
  ✓ كشف التعبئة بسيناريوهين (انتظار البيانات / بيانات فورية)
  ✓ max_tokens محدود لكل نوع
  ✓ LLM يقرر إذا كانت الرسالة تعبئة (لا قائمة كلمات)
"""

import os
from groq import Groq
from dotenv import load_dotenv
from retrieval import HybridRetriever

load_dotenv()

# ── الإعداد ───────────────────────────────────────────────────────────────
from config import GROQ_API_KEY, MODEL_FAST, MODEL_LEGAL
_client   = Groq(api_key=GROQ_API_KEY)
_retriever = HybridRetriever()

# النماذج محمّلة من config.py

# الشخصية الثابتة — قصيرة ومركّزة
_PERSONA = (
    "أنت مستشار قانوني ذكي للمحامين السوريين. "
    "متخصص في قانون الأحوال الشخصية وأصول المحاكمات. "
    "تتكلم كزميل خبير — مباشر، دقيق، بدون حشو."
)


# ══════════════════════════════════════════════════════════════════════════
# أداة مساعدة: استدعاء LLM
# ══════════════════════════════════════════════════════════════════════════

def _call(
    system: str,
    query: str,
    history: list,
    model: str = MODEL_FAST,
    max_tokens: int = 800,
    temperature: float = 0.1,
) -> str:
    """استدعاء LLM مع history مرشّح"""
    messages = [{"role": "system", "content": system}]
    for turn in (history or []):
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_stream(system, query, history, model=MODEL_LEGAL, max_tokens=800, temperature=0.1):
    """استدعاء LLM مع stream=True - يرجع generator"""
    messages = [{"role": "system", "content": system}]
    for turn in (history or []):
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    stream = _client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content



# ══════════════════════════════════════════════════════════════════════════
# كشف السيناريو — هل هذه تعبئة لقالب سابق؟
# ══════════════════════════════════════════════════════════════════════════

def _last_assistant_msg(history: list) -> str:
    """آخر رد من الـ assistant في الـ history"""
    for turn in reversed(history or []):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""

def _is_fill_request(query: str, history: list) -> bool:
    """
    يسأل LLM نفسه: هل هذه الرسالة تعبئة لقالب سبق عرضه؟
    يُستخدم عندما last_template موجود في الـ session.
    استدعاء واحد سريع بـ 8b.
    """
    last = _last_assistant_msg(history)
    if not last:
        return False

    prompt = f"""الرد السابق من النظام:
{last[:400]}

رسالة المحامي الجديدة:
{query}

سؤال واحد فقط: هل رسالة المحامي هي تزويد ببيانات لتعبئة الصياغة التي عُرضت أعلاه؟
أجب بـ "نعم" أو "لا" فقط."""

    answer = _call(
        system="أجب بكلمة واحدة فقط: نعم أو لا.",
        query=prompt,
        history=[],
        model=MODEL_FAST,
        max_tokens=5,
        temperature=0,
    )
    return "نعم" in answer


def _has_direct_client_data(query: str) -> bool:
    """
    السيناريو 2: المحامي أرسل بيانات موكل في نفس طلب الصياغة.
    مثال: "بدي صياغة طلاق للمدعية سارة أحمد ضد أحمد محمود"
    نستخدم LLM لأن الأسماء والبيانات لا تُحصى بقاموس.
    """
    prompt = f"""رسالة المحامي: {query}

هل تحتوي هذه الرسالة على بيانات موكل محددة مثل أسماء أشخاص أو أرقام أو تواريخ أو عنوان؟
أجب بـ "نعم" أو "لا" فقط."""

    answer = _call(
        system="أجب بكلمة واحدة فقط: نعم أو لا.",
        query=prompt,
        history=[],
        model=MODEL_FAST,
        max_tokens=5,
        temperature=0,
    )
    return "نعم" in answer


# ══════════════════════════════════════════════════════════════════════════
# Handler 0: تعبئة قالب (السيناريوهان 1 و 2)
# ══════════════════════════════════════════════════════════════════════════

def _handle_fill(
    query: str,
    template: dict,
    history: list,
) -> dict:
    """
    يعبّئ القالب المحفوظ ببيانات الموكل.
    يُستخدم في:
      - السيناريو 1: المحامي أرسل البيانات بعد عرض القالب
      - السيناريو 2: المحامي أرسل البيانات مع طلب الصياغة مباشرة
    """
    title   = template.get("title", "")
    text    = template.get("formal_text", "")
    notes   = template.get("intro_notes", "")

    system = f"""{_PERSONA}

لديك القالب القانوني التالي:
العنوان: {title}
{("ملاحظة: " + notes) if notes else ""}

نص القالب:
{text}

مهمتك: عبّئ هذا القالب ببيانات الموكل من رسالة المحامي.
- استبدل كل الفراغات (.....) بالبيانات المقدمة
- البيانات الناقصة اكتب مكانها [يرجى التزويد]
- لا تضف معلومات من عندك
- قدّم العريضة النهائية جاهزة للطباعة بدون مقدمات"""

    # نأخذ آخر 3 أدوار فقط — كافية للسياق
    recent = (history or [])[-6:]
    message = _call(
        system=system,
        query=query,
        history=recent,
        model=MODEL_FAST,
        max_tokens=1500,
        temperature=0.05,
    )

    # is_filled = True يخبر الـ frontend أن يعرض الـ message داخل template card
    filled_template = {**template, "is_filled": True}
    return {
        "message":     message,
        "template":    filled_template,
        "articles":    [],
        "attachments": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# Handler 1: عرض القالب (TEMPLATE بدون بيانات)
# ══════════════════════════════════════════════════════════════════════════

def _handle_template(query: str, history: list) -> dict:
    """
    يسترجع القالب المناسب ويعرضه.
    في نهاية الرد يسأل سؤالاً واحداً عن التعبئة.
    """
    rag = _retriever.get_template(query)
    if "error" in rag:
        return {
            "message":     "لم أجد صياغة مناسبة لهذا الطلب.",
            "template":    None,
            "articles":    [],
            "attachments": None,
        }

    title   = rag.get("title", "")
    text    = rag.get("formal_text", "")
    notes   = rag.get("intro_notes", "")
    # استخرج الحقول الفارغة من القالب لسؤال دقيق
    import re
    blanks  = re.findall(r'\.{3,}|_+|\(\.+\)', text)
    fields  = f"{len(blanks)} حقلاً" if blanks else "بعض الحقول"

    system = f"""{_PERSONA}

عرض صياغة قانونية جاهزة للمحامي.
القاعدة الوحيدة: اعرض القالب كما هو بدون تعديل أو إضافة.
{("ملاحظة مهنية قبل القالب: " + notes) if notes else ""}
في نهاية الرد اسأل سطراً واحداً فقط:
هل تريد التعبئة؟ إذا نعم زودني بـ: [{fields} المطلوبة]"""

    message = _call(
        system=system,
        query=f"اعرض القالب التالي:\nالعنوان: {title}\n\n{text}",
        history=[],          # لا history هنا — القالب واضح
        model=MODEL_FAST,
        max_tokens=1500,
        temperature=0.05,
    )

    return {
        "message":     message,
        "template":    rag,
        "articles":    [],
        "attachments": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# Handler 2: المرفقات (ATTACHMENT)
# ══════════════════════════════════════════════════════════════════════════

def _handle_attachment(query: str, history: list) -> dict:
    """يسترجع المرفقات ويقدمها بأسلوب مهني."""
    rag = _retriever.get_attachments(query)
    if "error" in rag:
        return {
            "message":     "لم أجد مرفقات لهذا الطلب.",
            "template":    None,
            "articles":    [],
            "attachments": rag,
        }

    atts  = rag.get("attachments", [])
    title = rag.get("title", "")
    atts_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(atts))

    system = f"""{_PERSONA}

قدّم المرفقات المطلوبة بأسلوب مهني ومباشر.
لا تكرر أي مرفق. لا تضف مرفقات من عندك."""

    message = _call(
        system=system,
        query=f"المرفقات المطلوبة لـ {title}:\n{atts_text}",
        history=[],
        model=MODEL_FAST,
        max_tokens=400,
        temperature=0.05,
    )

    return {
        "message":     message,
        "template":    None,
        "articles":    [],
        "attachments": rag,
    }


# ══════════════════════════════════════════════════════════════════════════
# Handler 3: السؤال القانوني (LEGAL_Q)
# ══════════════════════════════════════════════════════════════════════════

def _handle_legal(query: str, history: list, stream: bool = False):
    """
    يجيب على السؤال القانوني.
    stream=True  -> يرجع (generator, articles)
    stream=False -> يرجع dict
    """
    rag = _retriever.answer_legal_question(query)
    if "error" in rag:
        if stream:
            def _empty():
                yield "لم أجد مواد قانونية ذات صلة بهذا السؤال."
            return _empty(), []
        return {
            "message":     "لم أجد مواد قانونية ذات صلة بهذا السؤال.",
            "template":    None,
            "articles":    [],
            "attachments": None,
        }

    context  = rag.get("context", "")
    articles = rag.get("articles", [])
    recent   = (history or [])[-8:]

    system = f"""{_PERSONA}

السياق القانوني (من المواد المسترجعة فقط):
{context}

قواعد الإجابة:
- أجب فقط مما في السياق أعلاه
- اذكر رقم المادة داخل كلامك عند الاستشهاد
- لا تخترع شروطاً أو أحكاماً
- إذا لم تجد الإجابة في السياق صرّح بذلك بصدق
- تكلم كمستشار يشرح لزميله لا كقاموس"""

    if stream:
        gen = _call_stream(system=system, query=query, history=recent,
                           model=MODEL_LEGAL, max_tokens=800, temperature=0.1)
        return gen, articles

    message = _call(system=system, query=query, history=recent,
                    model=MODEL_LEGAL, max_tokens=800, temperature=0.1)
    message = _append_sources(message, articles)
    return {
        "message":     message,
        "template":    None,
        "articles":    articles,
        "attachments": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# Handler 4: الدردشة (CHAT)
# ══════════════════════════════════════════════════════════════════════════

def _handle_chat(query: str, history: list) -> dict:
    """يرد على الدردشة العامة."""
    recent = (history or [])[-4:]

    system = f"""{_PERSONA}

أجب بأسلوب طبيعي ومختصر.
إذا كان السؤال قانونياً لكن غامضاً اسأل سؤالاً واحداً للتوضيح.
إذا كان خارج اختصاصك وضّح نطاق عملك بلطف."""

    message = _call(
        system=system,
        query=query,
        history=recent,
        model=MODEL_FAST,
        max_tokens=300,
        temperature=0.2,
    )

    return {
        "message":     message,
        "template":    None,
        "articles":    [],
        "attachments": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# مصادر منظمة
# ══════════════════════════════════════════════════════════════════════════

def _append_sources(message: str, articles: list) -> str:
    """يضيف قسم المصادر في نهاية الإجابة القانونية — مرة واحدة فقط"""
    if not articles or "المصادر" in message or "المراجع" in message:
        return message

    seen, lines = set(), []
    for a in articles:
        art = a.get("article", "")
        law = a.get("law", "")
        key = f"{art}_{law}"
        if art and key not in seen:
            seen.add(key)
            lines.append(f"• {art}" + (f" — {law}" if law else ""))

    if not lines:
        return message

    return message + "\n\n---\n**المصادر:** " + " | ".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# دمج نتائج multi-intent
# ══════════════════════════════════════════════════════════════════════════

def _merge(results: list) -> dict:
    """
    يدمج نتائج عدة handlers في رد واحد.
    الترتيب الثابت: صياغة → مرفقات → قانوني
    كل قسم مرة واحدة فقط — لا تكرار.
    """
    if len(results) == 1:
        return results[0]

    parts     = []
    articles  = []
    template  = None
    attachments = None

    for r in results:
        if r.get("message"):
            parts.append(r["message"])
        if r.get("articles"):
            articles = r["articles"]
        if r.get("template"):
            template = r["template"]
        if r.get("attachments"):
            attachments = r["attachments"]

    return {
        "message":     "\n\n---\n\n".join(parts),
        "template":    template,
        "articles":    articles,
        "attachments": attachments,
    }


# ══════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية المُصدَّرة
# ══════════════════════════════════════════════════════════════════════════

def process(
    query: str,
    intents: list,
    history: list = None,
    last_template: dict = None,
) -> dict:
    """
    المدخلات:
      query         — سؤال المحامي
      intents       — من intent_classifier  مثل ["TEMPLATE"] أو ["TEMPLATE","LEGAL_Q"]
      history       — تاريخ المحادثة
      last_template — آخر قالب عُرض (محفوظ في session)

    المخرج:
      { message, template, articles, attachments, intents }
    """
    history  = history or []
    has_tmpl = "TEMPLATE"   in intents
    has_att  = "ATTACHMENT" in intents
    has_leg  = "LEGAL_Q"    in intents

    # ══════════════════════════════════════════════════════
    # السيناريو 1: المحامي يكمل تعبئة قالب سبق عرضه
    # ══════════════════════════════════════════════════════
    if last_template and history:
        if _is_fill_request(query, history):
            result = _handle_fill(query, last_template, history)
            return {**result, "intents": ["TEMPLATE"]}

    # ══════════════════════════════════════════════════════
    # السيناريو 2: طلب صياغة مع بيانات موكل فورية
    # ══════════════════════════════════════════════════════
    if has_tmpl and _has_direct_client_data(query):
        # نسترجع القالب أولاً ثم نعبّئه مباشرة
        rag = _retriever.get_template(query)
        if "error" not in rag:
            result = _handle_fill(query, rag, history)
            return {**result, "intents": ["TEMPLATE"]}

    # ══════════════════════════════════════════════════════
    # معالجة عادية — كل intent بـ handler مستقل
    # ══════════════════════════════════════════════════════
    results = []

    if has_tmpl:
        results.append(_handle_template(query, history))

    if has_att:
        results.append(_handle_attachment(query, history))

    if has_leg:
        results.append(_handle_legal(query, history))

    if not results:
        # CHAT أو intent غير معروف
        results.append(_handle_chat(query, history))

    result = _merge(results)
    return {**result, "intents": intents}


def process_stream(query, intents, history=None, last_template=None):
    """
    Streaming للـ LEGAL_Q النقي فقط.
    يرجع (meta, generator) او (None, None)
    """
    history  = history or []
    has_leg  = "LEGAL_Q"    in intents
    has_tmpl = "TEMPLATE"   in intents
    has_att  = "ATTACHMENT" in intents

    if has_leg and not has_tmpl and not has_att:
        gen, articles = _handle_legal(query, history, stream=True)
        meta = {"articles": articles, "template": None, "attachments": None}
        return meta, gen

    return None, None
