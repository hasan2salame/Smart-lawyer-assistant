#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04c_upload_fonon_v2.py
رفع فنون المحاماة — النسخة المُصلحة

python 04c_upload_fonon_v2.py
"""

import json, time, uuid, os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

INPUT_FILE      = "fonon_data.json"
COLLECTION_NAME = "legal_fonon"
BATCH_SIZE      = 39
SLEEP_SECS      = 35

Settings.embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
)
Settings.llm           = None
Settings.chunk_size    = 8192
Settings.chunk_overlap = 0

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)
    print(f"🗑  تم حذف: {COLLECTION_NAME}")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
print(f"✓ تم إنشاء: {COLLECTION_NAME}")

with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)
print(f"← {len(data)} قسم")

docs = []
for i, item in enumerate(data):
    meta           = item["metadata"]
    item_id        = item.get("id") or f"fonon_{item.get('template_id', i)}"
    attachments_str = " | ".join(item.get("attachments", []))

    # fonon_data_complete.json يستخدم "text" للصياغة
    # fonon_data.json (من extract) يستخدم "formal_text"
    # نقرأ كليهما مع أولوية لـ "text"
    formal_text = item.get("text") or item.get("formal_text", "")
    intro_notes = item.get("notes") or item.get("intro_notes", "")

    embedding_text = (
        f"{item['title']}\n"
        f"{formal_text}"
    ).strip()

    docs.append(Document(
        doc_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"fonon_v2_{item_id}")),
        text=embedding_text,
        metadata={
            "source":       "fonon",
            "id":           item_id,
            "template_id":  item.get("template_id", i),
            "title":        item["title"],
            "category":     meta.get("category", ""),
            "has_formal":   bool(formal_text),
            "attachments":  attachments_str,
            "formal_text":  formal_text,
            "intro_notes":  intro_notes,
            "post_notes":   item.get("post_notes", ""),
        },
        excluded_embed_metadata_keys=[
            "source","id","template_id","category","has_formal",
            "attachments","formal_text","intro_notes","post_notes"
        ],
        excluded_llm_metadata_keys=["source","has_formal"],
    ))

vector_store    = QdrantVectorStore(client=qdrant, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

total         = len(docs)
uploaded      = 0
total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, total, BATCH_SIZE):
    batch     = docs[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1

    for attempt in range(3):
        try:
            VectorStoreIndex.from_documents(
                batch, storage_context=storage_context, show_progress=False,
            )
            uploaded += len(batch)
            info = qdrant.get_collection(COLLECTION_NAME)
            print(f"  ✓ دفعة {batch_num}/{total_batches} — {uploaded}/{total} — نقاط: {info.points_count}")
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  ⚠ محاولة {attempt+1}: {str(e)[:100]} — انتظار {wait}ث")
            time.sleep(wait)
    else:
        print(f"  ✗ تخطي دفعة {batch_num}")

    if i + BATCH_SIZE < total:
        time.sleep(SLEEP_SECS)

info = qdrant.get_collection(COLLECTION_NAME)
print(f"\n✅ legal_fonon — {info.points_count} نقطة")
if info.points_count == total:
    print("   ✅ مثالي — كل صياغة = نقطة واحدة")