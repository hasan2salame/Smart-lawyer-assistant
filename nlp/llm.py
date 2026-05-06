"""
nlp/llm.py
LLM response layer — four independent handlers, one per intent class.

Architecture
------------
[1] Scenario detection — template fill (one unified check, not two separate calls)
[2] Each intent handled independently to prevent context contamination.
[3] Results merged with fixed ordering: template → attachments → legal.

Design decisions
----------------
- Handlers are independent: no shared state between intents in one request.
- Out-of-scope guard: when retrieval confidence < threshold, the system
  returns a scoped refusal instead of hallucinating.
- Fill detection is a single lightweight binary LLM call (8b, 5 tokens).
- max_tokens is bounded per handler to control cost and latency.
"""

import re

from groq import Groq

from config import GROQ_API_KEY, MODEL_FAST, MODEL_LEGAL
from retrieval import HybridRetriever

_groq      = Groq(api_key=GROQ_API_KEY)
_retriever = HybridRetriever()

_PERSONA = (
    "أنت مستشار قانوني ذكي للمحامين السوريين. "
    "متخصص في قانون الأحوال الشخصية وأصول المحاكمات. "
    "تتكلم كزميل خبير — مباشر، دقيق، بدون حشو."
)

# Token budget per intent type.
# Multi-intent requests accumulate budgets so each part gets its full space.
# Single-intent: TEMPLATE ~800, LEGAL_Q ~1000, ATTACHMENT ~200.
# Cap at 4096 — safe upper bound for llama-3.3-70b context window usage.
_TOKENS_PER_INTENT: dict[str, int] = {
    "TEMPLATE":   900,
    "LEGAL_Q":    1100,
    "ATTACHMENT": 250,
    "CHAT":       300,
}
_MAX_TOKENS_CAP = 4096


def _budget(intents: list[str]) -> int:
    """
    Compute dynamic max_tokens for the full response based on active intents.

    Single intent  → that intent's budget (e.g. LEGAL_Q = 1100)
    Multi-intent   → sum of all active budgets, capped at _MAX_TOKENS_CAP
    """
    total = sum(_TOKENS_PER_INTENT.get(i, 500) for i in intents)
    return min(total, _MAX_TOKENS_CAP)

# Base message for out-of-scope queries — LLM may rephrase it naturally
# but the core meaning must be preserved: specialist system, rephrase your question.
_OUT_OF_SCOPE_BASE = (
    "انا نظام مساعد ذكي متخصص في مجال القانون "
    "لذلك اعد صياغة سؤالك في هذا المجال من فضلك بشكل ادق "
    "لتحصل على اجابة 😊"
)

# ── LLM call helpers ──────────────────────────────────────────────────────

def _call(
    system: str,
    query: str,
    history: list,
    model: str       = MODEL_FAST,
    max_tokens: int  = 800,
    temperature: float = 0.1,
) -> str:
    """Synchronous LLM call with optional conversation history."""
    messages = [{"role": "system", "content": system}]
    for turn in (history or []):
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    resp = _groq.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_stream(
    system: str,
    query: str,
    history: list,
    model: str       = MODEL_LEGAL,
    max_tokens: int  = 800,
    temperature: float = 0.1,
):
    """Streaming LLM call — yields text tokens."""
    messages = [{"role": "system", "content": system}]
    for turn in (history or []):
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    stream = _groq.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ── Scenario detection ────────────────────────────────────────────────────

def _last_assistant_msg(history: list) -> str:
    """Return the most recent assistant message from history."""
    for turn in reversed(history or []):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def _detect_fill_scenario(
    query: str,
    history: list,
    last_template: dict | None,
) -> str:
    """
    Determine which fill scenario applies, using a single LLM call.

    Returns
    -------
    "fill_history"  — lawyer is providing data for a previously shown template
    "fill_inline"   — lawyer embedded client data in this drafting request
    "none"          — no fill scenario detected
    """
    has_tmpl  = bool(last_template and history)
    last_msg  = _last_assistant_msg(history) if has_tmpl else ""

    prompt_parts = []
    if has_tmpl:
        prompt_parts.append(
            f"الرد السابق من النظام:\n{last_msg[:400]}\n\n"
            f"رسالة المحامي الجديدة:\n{query}\n\n"
            "هل رسالة المحامي هي تزويد ببيانات لتعبئة صياغة سابقة (أ)، "
            "أم تحتوي على بيانات موكل ضمن طلب جديد (ب)، أم لا شيء من ذلك (ج)؟ "
            "أجب بحرف واحد فقط: أ أو ب أو ج."
        )
    else:
        prompt_parts.append(
            f"رسالة المحامي: {query}\n\n"
            "هل تحتوي على بيانات موكل محددة (أسماء / أرقام / تواريخ)؟ "
            "أجب بـ: ب إذا نعم، ج إذا لا."
        )

    answer = _call(
        system="أجب بحرف واحد فقط.",
        query=prompt_parts[0],
        history=[],
        model=MODEL_FAST,
        max_tokens=5,
        temperature=0,
    ).strip()

    if "أ" in answer:
        return "fill_history"
    if "ب" in answer:
        return "fill_inline"
    return "none"


# ── Handler 0 — Template fill ─────────────────────────────────────────────

def _handle_fill(query: str, template: dict, history: list, max_tokens: int = 2048) -> dict:
    """
    Fill a stored template with client data.

    Covers two scenarios:
    - Scenario 1: lawyer provides data after the template was displayed.
    - Scenario 2: lawyer embedded client data in the same drafting request.
    """
    title = template.get("title", "")
    text  = template.get("formal_text", "")
    notes = template.get("intro_notes", "")

    system = (
        f"{_PERSONA}\n\n"
        f"لديك القالب القانوني التالي:\nالعنوان: {title}\n"
        f"{'ملاحظة: ' + notes if notes else ''}\n\n"
        f"نص القالب:\n{text}\n\n"
        "مهمتك: عبّئ هذا القالب ببيانات الموكل من رسالة المحامي.\n"
        "- استبدل كل الفراغات (.....) بالبيانات المقدمة\n"
        "- البيانات الناقصة اكتب مكانها [يرجى التزويد]\n"
        "- لا تضف معلومات من عندك\n"
        "- قدّم العريضة النهائية جاهزة للطباعة بدون مقدمات"
    )
    message = _call(
        system=system,
        query=query,
        history=(history or [])[-6:],
        model=MODEL_FAST,
        max_tokens=max_tokens,
        temperature=0.05,
    )
    return {
        "message":     message,
        "template":    {**template, "is_filled": True},
        "articles":    [],
        "attachments": None,
    }


# ── Handler 1 — Template display ─────────────────────────────────────────

def _handle_template(query: str, history: list, max_tokens: int = 2048) -> dict:
    """
    Retrieve the best-matching template and display it verbatim.
    Ends with a single question asking whether the lawyer wants to fill it.
    """
    rag = _retriever.get_template(query)
    if "error" in rag:
        return {
            "message":     "لم أجد صياغة مناسبة لهذا الطلب.",
            "template":    None,
            "articles":    [],
            "attachments": None,
        }

    title  = rag.get("title", "")
    text   = rag.get("formal_text", "")
    notes  = rag.get("intro_notes", "")
    blanks = re.findall(r"\.{3,}|_+|\(\.\+\)", text)
    fields = f"{len(blanks)} حقلاً" if blanks else "بعض الحقول"

    system = (
        f"{_PERSONA}\n\n"
        "عرض صياغة قانونية جاهزة للمحامي.\n"
        "القاعدة الوحيدة: اعرض القالب كما هو بدون تعديل أو إضافة.\n"
        f"{'ملاحظة مهنية قبل القالب: ' + notes if notes else ''}\n"
        f"في نهاية الرد اسأل سطراً واحداً فقط:\n"
        f"هل تريد التعبئة؟ إذا نعم زودني بـ: [{fields} المطلوبة]"
    )
    message = _call(
        system=system,
        query=f"اعرض القالب التالي:\nالعنوان: {title}\n\n{text}",
        history=[],
        model=MODEL_FAST,
        max_tokens=max_tokens,
        temperature=0.05,
    )
    return {
        "message":     message,
        "template":    rag,
        "articles":    [],
        "attachments": None,
    }


# ── Handler 2 — Attachments ───────────────────────────────────────────────

def _handle_attachment(query: str, history: list) -> dict:
    """Retrieve and present required court documents professionally."""
    rag = _retriever.get_attachments(query)
    if "error" in rag:
        return {
            "message":     "لم أجد مرفقات لهذا الطلب.",
            "template":    None,
            "articles":    [],
            "attachments": rag,
        }

    title     = rag.get("title", "")
    atts      = rag.get("attachments", [])
    atts_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(atts))

    system = (
        f"{_PERSONA}\n\n"
        "قدّم المرفقات المطلوبة بأسلوب مهني ومباشر.\n"
        "لا تكرر أي مرفق. لا تضف مرفقات من عندك."
    )
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


# ── Handler 3 — Legal Q&A ─────────────────────────────────────────────────

def _handle_legal(
    query: str,
    history: list,
    stream: bool = False,
    max_tokens: int = 2048,
):
    """
    Answer a legal question from retrieved statutory articles.

    Out-of-scope guard
    ------------------
    If retrieval returns {"error": "out_of_scope"} (top Cohere Rerank score
    < CONFIDENCE_THRESHOLD), returns a polite scoped refusal — never hallucinates.

    Parameters
    ----------
    stream : bool
        True  → returns (generator, articles)
        False → returns dict
    """
    rag = _retriever.answer_legal_question(query)

    if "error" in rag:
        # Static refusal — never route through LLM.
        # If LLM receives the query in refusal context, it answers from its
        # own training knowledge, which causes hallucination.
        if stream:
            def _refused():
                yield _OUT_OF_SCOPE_BASE
            return _refused(), []
        return {
            "message":     _OUT_OF_SCOPE_BASE,
            "template":    None,
            "articles":    [],
            "attachments": None,
        }

    context  = rag.get("context", "")
    articles = rag.get("articles", [])
    recent   = (history or [])[-8:]

    system = (
        # Anti-hallucination rules come FIRST — before persona — so the model
        # reads them before any context that might trigger self-introduction.
        "⚠ قواعد مطلقة لا استثناء فيها:\n"
        "- لا تبدأ ردّك بـ 'شكرًا' أو 'كمتخصص' أو 'أود أن أُشير' أو أي مقدمة.\n"
        "- ابدأ فوراً بالإجابة الفعلية — أول كلمة في ردك يجب أن تكون من صلب الجواب.\n"
        "- أجب فقط مما في المواد القانونية أدناه. معرفتك الخاصة غير موجودة هنا.\n"
        "- كل حكم أو شرط تذكره يجب أن يُنسب لمادة قانونية برقمها.\n"
        "- إذا لم تجد الجواب في المواد أدناه قل بالضبط: "
        "'لا تتوفر في المواد المسترجعة معلومات كافية حول هذه النقطة.'\n"
        "\n"
        f"{_PERSONA}\n\n"
        f"المواد القانونية المسترجعة:\n{context}\n\n"
        "اكتب الإجابة كاملة ومفصّلة — المحامي يحتاج كل التفاصيل."
    )

    if stream:
        gen = _call_stream(
            system=system,
            query=query,
            history=recent,
            model=MODEL_LEGAL,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return gen, articles

    message = _call(
        system=system,
        query=query,
        history=recent,
        model=MODEL_LEGAL,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return {
        "message":     _append_sources(message, articles),
        "template":    None,
        "articles":    articles,
        "attachments": None,
    }


# ── Handler 4 — Chat ──────────────────────────────────────────────────────

def _handle_chat(query: str, history: list) -> dict:
    """
    Handle greetings, follow-up questions, and conversational messages.

    Scope rules
    -----------
    - Greetings, thanks, short follow-ups → reply naturally.
    - Ambiguous legal questions → ask one clarifying question.
    - Clearly off-topic (math, weather, sports, science...) → politely refuse
      using _OUT_OF_SCOPE_BASE, do NOT answer the question.
    """
    system = (
        f"{_PERSONA}\n\n"
        "قواعد صارمة:\n"
        "1. التحية والشكر والمتابعة القصيرة → أجب بأسلوب طبيعي ومختصر بدون مقدمات.\n"
        "2. سؤال قانوني غامض → اسأل سؤالاً توضيحياً واحداً فقط — لا تُعرّف بنفسك.\n"
        "3. أي سؤال خارج نطاق الأحوال الشخصية وأصول المحاكمات تماماً "
        "(رياضيات، طقس، علوم، تاريخ، جغرافيا، تقنية، وأي موضوع غير قانوني) → "
        "أجب فقط بهذه العبارة بالضبط دون إجابة السؤال: "
        f"\"{_OUT_OF_SCOPE_BASE}\""
        "\n4. لا تبدأ أي رد بـ 'شكرًا' أو 'أود أن أُشير' أو أي جملة تعريفية."
    )
    message = _call(
        system=system,
        query=query,
        history=(history or [])[-4:],
        model=MODEL_FAST,
        max_tokens=300,
        temperature=0.1,
    )
    return {
        "message":     message,
        "template":    None,
        "articles":    [],
        "attachments": None,
    }


# ── Source citation ───────────────────────────────────────────────────────

def _append_sources(message: str, articles: list) -> str:
    """Append a structured sources section to a legal QA response."""
    if not articles or "المصادر" in message or "المراجع" in message:
        return message

    seen:  set[str]  = set()
    lines: list[str] = []
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


# ── Multi-intent merge ────────────────────────────────────────────────────

def _merge(results: list[dict]) -> dict:
    """
    Merge multiple handler results into a single response.
    Fixed ordering: template → attachments → legal.
    """
    if len(results) == 1:
        return results[0]

    parts:       list[str]       = []
    articles:    list            = []
    template:    dict | None     = None
    attachments: dict | None     = None

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


# ── Public API ────────────────────────────────────────────────────────────

def process(
    query: str,
    intents: list[str],
    history: list        = None,
    last_template: dict  = None,
) -> dict:
    """
    Route a classified query to the appropriate handler(s) and return a
    merged response dict.

    Parameters
    ----------
    query         : lawyer's message
    intents       : from the classifier, e.g. ["TEMPLATE"] or ["LEGAL_Q"]
    history       : conversation history (list of {role, content} dicts)
    last_template : last template displayed in this session (for fill detection)

    Returns
    -------
    dict with keys: message, template, articles, attachments, intents
    """
    history  = history or []
    has_tmpl = "TEMPLATE"   in intents
    has_att  = "ATTACHMENT" in intents
    has_leg  = "LEGAL_Q"    in intents

    # Scenario detection — only when there is actual context to fill from.
    # Skipped when last_template is None AND history is empty: a plain new
    # drafting request cannot be a fill request, so the LLM call is wasteful.
    if has_tmpl and (last_template or history):
        scenario = _detect_fill_scenario(query, history, last_template)

        if scenario == "fill_history" and last_template:
            result = _handle_fill(query, last_template, history,
                                  max_tokens=_TOKENS_PER_INTENT["TEMPLATE"])
            return {**result, "intents": ["TEMPLATE"]}

        if scenario == "fill_inline":
            rag = _retriever.get_template(query)
            if "error" not in rag:
                result = _handle_fill(query, rag, history,
                                      max_tokens=_TOKENS_PER_INTENT["TEMPLATE"])
                return {**result, "intents": ["TEMPLATE"]}

    # Compute token budget dynamically based on the number of active intents.
    # Multi-intent requests get more tokens so each part is fully answered.
    active   = [i for i in intents if i in _TOKENS_PER_INTENT]
    mt_each  = {i: _TOKENS_PER_INTENT[i] for i in active}
    mt_total = min(sum(mt_each.values()), _MAX_TOKENS_CAP)

    # Normal processing — independent handler per intent
    results: list[dict] = []

    if has_tmpl:
        results.append(_handle_template(query, history, max_tokens=mt_each.get("TEMPLATE", 900)))
    if has_att:
        results.append(_handle_attachment(query, history))
    if has_leg:
        results.append(_handle_legal(query, history, max_tokens=mt_each.get("LEGAL_Q", 1100)))
    if not results:
        results.append(_handle_chat(query, history))

    result = _merge(results)
    return {**result, "intents": intents}


def process_stream(
    query: str,
    intents: list[str],
    history: list       = None,
    last_template: dict = None,
):
    """
    Streaming path — only for pure LEGAL_Q intent (no TEMPLATE or ATTACHMENT).

    Returns
    -------
    (meta_dict, token_generator) if streaming applies, else (None, None).
    """
    history  = history or []
    has_leg  = "LEGAL_Q"    in intents
    has_tmpl = "TEMPLATE"   in intents
    has_att  = "ATTACHMENT" in intents

    if has_leg and not has_tmpl and not has_att:
        gen, articles = _handle_legal(query, history, stream=True,
                                      max_tokens=_TOKENS_PER_INTENT["LEGAL_Q"])
        meta = {
            "articles":    articles,
            "template":    None,
            "attachments": None,
        }
        return meta, gen

    return None, None
