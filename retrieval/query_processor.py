"""
retrieval/query_processor.py
Text cleaning and query rewriting for Arabic legal queries.

clean_text  — normalisation only, no external calls (used everywhere).
rewrite_query — converts colloquial Syrian Arabic to formal MSA legal
                language via Groq.  Skipped when the query already
                contains an explicit article reference (e.g. "المادة 137").
"""

import re
from groq import Groq
from config import GROQ_API_KEY, MODEL_FAST

_groq = Groq(api_key=GROQ_API_KEY)

# Regex that matches explicit article references — rewrite is unnecessary
_ARTICLE_RE = re.compile(r"(المادة|مادة|م\.)\s*\d+", re.UNICODE)

_REWRITE_SYSTEM = """You are a Syrian legal language expert specialising in Personal Status Law and Civil Procedure Law.

Convert the user's colloquial or mixed-language query into a single concise formal Arabic legal question suitable for searching Syrian statutory law.

Critical rules
--------------
1. Preserve procedural terms precisely:
   - "تبليغ" → "إجراءات التبليغ القضائي" (judicial notification, NOT spousal notification)
   - "استئناف" → "إجراءات الاستئناف القضائي"
   - "اختصاص" → "الاختصاص القضائي للمحكمة"
   - "تنفيذ" → "تنفيذ الأحكام القضائية"

2. Do NOT drift into personal status topics when the query is procedural.
   "ما إجراءات تبليغ المدعى عليه" → STAYS procedural, do not convert to spousal context.

3. Keep article references unchanged (e.g. "المادة 137" must not be modified).

4. Return ONLY the rewritten question — no explanation, no preamble."""


def clean_text(text: str) -> str:
    """
    Normalise Arabic text for retrieval and display.

    Removes
    -------
    - Tashkeel (harakat) and tatweel (kashida)
    - Non-Arabic punctuation and special characters
    - Extra whitespace

    Normalises
    ----------
    - Hamza variants  → bare alef  (أإآ → ا)
    - Ta marbuta      → ha         (ة   → ه)
    - Alef maqsura    → ya         (ى   → ي)
    """
    if not text:
        return ""

    # Remove tashkeel (U+064B – U+065F) and tatweel (U+0640)
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)

    # Normalise hamza variants to bare alef
    text = re.sub(r"[أإآ]", "ا", text)

    # Normalise ta marbuta and alef maqsura
    text = text.replace("ة", "ه").replace("ى", "ي")

    # Remove anything that is not Arabic, space, digit, or basic punctuation
    text = re.sub(r"[^\u0600-\u06FF\s\d.,،؛:؟!()\"'-]", "", text)

    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def rewrite_query(query: str) -> str:
    """
    Rewrite a colloquial/mixed query to formal MSA legal language.

    Returns the original query unchanged when:
    - It already contains an explicit article reference, or
    - The Groq call fails for any reason.
    """
    if not query or _ARTICLE_RE.search(query):
        return query

    try:
        response = _groq.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user",   "content": query},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten or query
    except Exception:
        return query
