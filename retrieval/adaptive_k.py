"""
retrieval/adaptive_k.py
Pure gap-based dynamic Top-K selection.

بدل حد أدنى وأعلى مفروضَين، النظام يشوف التوزيع الفعلي للـ scores
ويوقف عند أكبر gap حقيقية بين أي عنصرين متجاورين.

النتيجة:
  - سؤال عن مادة واحدة محددة   → قد يُرجع 1 فقط
  - سؤال عن موضوع معقد متشعّب  → قد يُرجع 10 أو أكثر
  - كل العناصر ذات صلة متشابهة → يُرجع الكل (حتى ADAPTIVE_K_MAX)

الحد الأعلى ADAPTIVE_K_MAX موجود فقط لحماية context window الـ LLM،
وليس حكماً على صلة النتائج.
"""

from config import ADAPTIVE_K_MAX, ADAPTIVE_GAP_THRESHOLD


def adaptive_filter(items: list[dict]) -> list[dict]:
    """
    Return items above the largest score cliff.

    Steps
    -----
    1. إذا عنصر واحد أو فارغ → أرجعه مباشرة.
    2. احسب كل الـ gaps بين العناصر المتجاورة.
    3. طبّع الـ gaps على المدى الكلي (hi - lo) لمقارنة عادلة.
    4. إذا أكبر gap أصغر من ADAPTIVE_GAP_THRESHOLD → لا cliff حقيقي
       → أرجع الكل حتى ADAPTIVE_K_MAX.
    5. إذا في cliff واضح → قطع هناك.
    """
    if not items:
        return []

    if len(items) == 1:
        return items

    scores = [item["score"] for item in items]
    lo, hi = min(scores), max(scores)

    # كل العناصر بنفس الـ score → أرجع الكل حتى MAX
    if hi == lo:
        return items[:ADAPTIVE_K_MAX]

    # طبّع على [0, 1] لمقارنة الـ gaps بشكل مستقل عن مقياس الـ reranker
    normed = [(s - lo) / (hi - lo) for s in scores]
    gaps   = [normed[i] - normed[i + 1] for i in range(len(normed) - 1)]

    max_gap     = max(gaps)
    max_gap_idx = gaps.index(max_gap)

    # لا cliff حقيقي → كل العناصر ذات صلة متشابهة
    if max_gap < ADAPTIVE_GAP_THRESHOLD:
        return items[:ADAPTIVE_K_MAX]

    # قطع عند الـ cliff
    cutoff = max_gap_idx + 1
    return items[:min(cutoff, ADAPTIVE_K_MAX)]


def check_confidence(reranked: list[dict]) -> bool:
    """Return True if retrieval found at least one article."""
    return bool(reranked)
