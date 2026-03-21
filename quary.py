#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_retrieval.py — النسخة المُصلحة
كل مسار يعرض ما يناسبه فقط
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-multilingual-v3.0",
    input_type="search_query",
)
Settings.llm = None

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)

def get_retriever(collection, top_k=5):
    store = QdrantVectorStore(client=qdrant, collection_name=collection)
    index = VectorStoreIndex.from_vector_store(store)
    return index.as_retriever(similarity_top_k=top_k)

retriever_laws  = get_retriever("legal_laws")
retriever_osoul = get_retriever("legal_osoul")
retriever_fonon = get_retriever("legal_fonon")

print("✅ الاتصال بـ Qdrant نجح\n")

SEP = "═" * 60

# ══════════════════════════════════════════════════════════════
# مسار 1: صياغة — يعرض الصياغة الرسمية كاملة
# ══════════════════════════════════════════════════════════════
def test_template(query):
    print(SEP)
    print(f"  مسار 1 — صياغة")
    print(f"  الاستعلام: {query}")
    print(SEP)
    nodes = retriever_fonon.retrieve(query)
    best  = nodes[0]
    meta  = best.node.metadata
    print(f"\n  العنوان : {meta.get('title','')}")
    print(f"  التصنيف : {meta.get('category','')}")
    print(f"  score   : {round(best.score or 0, 3)}")
    print(f"\n  ── الصياغة الرسمية ──────────────────────────")
    print(meta.get('formal_text','— لا توجد صياغة رسمية —'))
    print()

# ══════════════════════════════════════════════════════════════
# مسار 2: مرفقات — المرفقات فقط بدون أي نص
# ══════════════════════════════════════════════════════════════
def test_attachments(query):
    print(SEP)
    print(f"  مسار 2 — مرفقات")
    print(f"  الاستعلام: {query}")
    print(SEP)
    nodes = retriever_fonon.retrieve(query)
    best  = nodes[0]
    meta  = best.node.metadata
    atts  = [a.strip() for a in meta.get("attachments","").split("|") if a.strip()]
    print(f"\n  العنوان : {meta.get('title','')}")
    print(f"  score   : {round(best.score or 0, 3)}")
    print(f"\n  ── المرفقات المطلوبة ────────────────────────")
    for i, att in enumerate(atts, 1):
        print(f"  {i}. {att}")
    print()

# ══════════════════════════════════════════════════════════════
# مسار 3: سؤال قانوني — المواد كاملة مع أرقامها
# ══════════════════════════════════════════════════════════════
def test_legal(query, collection="laws"):
    retriever  = retriever_laws if collection == "laws" else retriever_osoul
    law_label  = "أحوال شخصية" if collection == "laws" else "أصول محاكمات"
    print(SEP)
    print(f"  مسار 3 — سؤال قانوني ({law_label})")
    print(f"  الاستعلام: {query}")
    print(SEP)
    nodes = retriever.retrieve(query)
    for i, n in enumerate(nodes[:3], 1):
        meta  = n.node.metadata
        score = round(n.score or 0, 3)
        print(f"\n  [{i}] {meta.get('article_str','')} — score={score}")
        print(f"       الباب : {meta.get('bab','')[:70]}")
        print(f"  ── النص ──────────────────────────────────")
        print(f"  {meta.get('original_text','')}")
    print()

# ══════════════════════════════════════════════════════════════
test_template("صياغة دعوىة نشوز الزوجة؟")
test_attachments("ماهي مرفقات دعوى مشاهدة الاطفال؟")
test_legal("شروط الخطبة؟", "laws")
test_legal("ما هي شروط البيع ؟", "osoul")