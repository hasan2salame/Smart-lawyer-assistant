#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py
المنسّق الرئيسي — يربط المصنف بطبقة الاستجابة

المسار:
  query → classify() → process() → dict
"""

from nlp.classifier import classify
from nlp.llm import process


def ask(
    query: str,
    history: list = None,
    last_template: dict = None,
) -> dict:
    """
    query         — سؤال المحامي
    history       — تاريخ المحادثة (من الـ session)
    last_template — آخر قالب عُرض (من الـ session)

    المخرج:
    {
        intents:     list[str]
        message:     str
        results:     list[dict]
        _template:   dict | None   ← لحفظه في الـ session
    }
    """
    if not query or not query.strip():
        return {
            "intents":   ["CHAT"],
            "message":   "كيف يمكنني مساعدتك اليوم؟",
            "results":   [],
            "_template": None,
        }

    history = history or []

    # [1] تصنيف النية
    classification = classify(query)
    intents        = classification["intents"]

    # [2] معالجة
    result = process(
        query=query,
        intents=intents,
        history=history,
        last_template=last_template,
    )

    # [3] بناء المخرج
    return {
        "intents": result.get("intents", intents),
        "message": result["message"],
        "results": [{
            "intent":      result.get("intents", intents)[0] if len(result.get("intents", intents)) == 1 else "MIXED",
            "message":     result["message"],
            "articles":    result.get("articles", []),
            "template":    result.get("template"),
            "attachments": result.get("attachments"),
        }],
        "_template": result.get("template"),
    }
