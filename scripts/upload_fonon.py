#!/usr/bin/env python3
"""
scripts/upload_fonon.py
Upload legal drafting templates (fonon) to Qdrant.

Accepts two possible JSON schemas:
  fonon_data_complete.json — uses "text" for the template body
  fonon_data.json          — uses "formal_text" for the template body
Both are handled transparently; "text" takes priority.

Usage
-----
    python scripts/upload_fonon.py
    # or via Makefile:
    make upload-fonon
"""

import json
import time
import uuid
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY

INPUT_FILE      = "data/processed/fonon.json"
COLLECTION_NAME = "legal_fonon"
BATCH_SIZE      = 28        # small corpus — one batch is fine
SLEEP_SECS      = 35        # Cohere free-tier rate-limit buffer

Settings.embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
)
Settings.llm           = None
Settings.chunk_size    = 8192   # keeps each template as a single vector node
Settings.chunk_overlap = 0


def main() -> None:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    # Drop and recreate collection for a clean upload
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"Dropped: {COLLECTION_NAME}")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"Created: {COLLECTION_NAME}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} templates")

    docs: list[Document] = []
    for i, item in enumerate(data):
        meta            = item.get("metadata", {})
        item_id         = item.get("id") or f"fonon_{item.get('template_id', i)}"
        attachments_str = " | ".join(item.get("attachments", []))

        # "text" key takes priority over "formal_text"
        formal_text = item.get("text") or item.get("formal_text", "")
        intro_notes = item.get("notes") or item.get("intro_notes", "")

        embedding_text = f"{item['title']}\n{formal_text}".strip()

        docs.append(Document(
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"fonon_v2_{item_id}")),
            text=embedding_text,
            metadata={
                "source":      "fonon",
                "id":          item_id,
                "template_id": item.get("template_id", i),
                "title":       item["title"],
                "category":    meta.get("category", ""),
                "has_formal":  bool(formal_text),
                "attachments": attachments_str,
                "formal_text": formal_text,
                "intro_notes": intro_notes,
                "post_notes":  item.get("post_notes", ""),
            },
            excluded_embed_metadata_keys=[
                "source", "id", "template_id", "category", "has_formal",
                "attachments", "formal_text", "intro_notes", "post_notes",
            ],
            excluded_llm_metadata_keys=["source", "has_formal"],
        ))

    vector_store    = QdrantVectorStore(client=qdrant, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    total         = len(docs)
    uploaded      = 0
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total, BATCH_SIZE):
        batch     = docs[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(3):
            try:
                VectorStoreIndex.from_documents(
                    batch, storage_context=storage_context, show_progress=False,
                )
                uploaded += len(batch)
                info = qdrant.get_collection(COLLECTION_NAME)
                print(f"  batch {batch_num}/{total_batches} "
                      f"— {uploaded}/{total} uploaded "
                      f"({info.points_count} in collection)")
                break
            except Exception as exc:
                wait = 15 * (attempt + 1)
                print(f"  attempt {attempt + 1}: {str(exc)[:100]} — retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  skipping batch {batch_num} after 3 failed attempts")

        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_SECS)

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\nlegal_fonon — {info.points_count} points")
    if info.points_count == total:
        print("  OK — one point per template")
    else:
        print(f"  WARNING: {total - info.points_count} templates missing — re-run script")


if __name__ == "__main__":
    main()
