"""
pipeline.py
Main orchestrator — connects the intent classifier to the LLM response layer.

Flow:  query → classify() [if intents not provided] → process() → unified response dict

Design decision
---------------
ask() accepts an optional `intents` parameter.  When provided (e.g. from a
streaming path that already classified the query), classify() is skipped entirely
— one API call saved per non-LEGAL_Q streaming request.
"""

from nlp.classifier import classify
from nlp.llm        import process


def ask(
    query: str,
    history: list        = None,
    last_template: dict  = None,
    intents: list[str]   = None,
) -> dict:
    """
    Classify a lawyer's query and route it to the appropriate handler(s).

    Parameters
    ----------
    query         : lawyer's message
    history       : conversation history (list of {role, content} dicts)
    last_template : last displayed template stored in the session
    intents       : pre-computed intents — skips classify() when provided

    Returns
    -------
    dict with keys:
        intents   : list[str]    — detected intent(s)
        message   : str          — final response text
        results   : list[dict]   — structured per-intent results
        _template : dict | None  — template to persist in session state
    """
    if not query or not query.strip():
        return {
            "intents":   ["CHAT"],
            "message":   "كيف يمكنني مساعدتك اليوم؟",
            "results":   [],
            "_template": None,
        }

    history = history or []

    # Classify only when intents are not already known
    if intents is None:
        intents = classify(query)["intents"]

    result = process(
        query=query,
        intents=intents,
        history=history,
        last_template=last_template,
    )

    final_intents = result.get("intents", intents)
    intent_label  = final_intents[0] if len(final_intents) == 1 else "MIXED"

    return {
        "intents": final_intents,
        "message": result["message"],
        "results": [
            {
                "intent":      intent_label,
                "message":     result["message"],
                "articles":    result.get("articles", []),
                "template":    result.get("template"),
                "attachments": result.get("attachments"),
            }
        ],
        "_template": result.get("template"),
    }
