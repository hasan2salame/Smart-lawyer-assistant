# Syrian Legal AI Assistant

A production-grade Hybrid RAG system for Syrian lawyers, specialising in
Personal Status Law and Civil Procedure Law.

---

## What It Does

| Intent | Example Input | Output |
|--------|--------------|--------|
| **TEMPLATE** | "Draft a divorce petition" | Ready-to-print court template + auto-fill |
| **ATTACHMENT** | "What documents do I need for a custody case?" | Required court documents list |
| **LEGAL_Q** | "What are the conditions for child custody?" | Answer grounded in statutory articles |
| **CHAT** | "Hello" | Conversational reply |

---

## Architecture

```
Lawyer's message
      │
      ▼
┌─────────────────────┐
│  Intent Classifier  │  Layer 1: keyword rules (0 ms, 0 API cost)
│  classifier.py      │  Layer 2: Groq LLaMA-3.1-8b (ambiguous queries only)
└────────┬────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│              Retrieval Layer                   │
│                                                │
│  Path 1 (TEMPLATE / ATTACHMENT)                │
│    Rerank-all over 28 fonon documents          │
│    (cross-encoder beats dense on tiny corpus)  │
│                                                │
│  Path 2 (LEGAL_Q) — True Hybrid RAG            │
│    Dense(laws) ──┐                             │
│    Dense(osoul) ─┤                             │
│                  ├─► RRF ─► Graph ─► Rerank   │
│    BM25(laws)  ──┤                             │
│    BM25(osoul) ──┘                             │
│    → Adaptive K (elbow method)                 │
│    → Out-of-scope: empty retrieval → refusal (no fixed score threshold)       │
└────────┬───────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│     LLM Layer       │  Four independent handlers (one per intent)
│     llm.py          │  Groq LLaMA-3.3-70b — true SSE streaming
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FastAPI + SQLite   │  Persistent sessions, multi-window support
│  main.py            │
└─────────────────────┘
```

---

## Data

| Qdrant Collection | Content | Size |
|-------------------|---------|------|
| `legal_laws` | Syrian Personal Status Law 1953 | 308 articles |
| `legal_osoul` | Syrian Civil Procedure Law 2016 | 495 articles |
| `legal_fonon` | Legal drafting templates (PDF) | 28 templates |

**Knowledge graph:** 831 nodes — 2,884 edges

Edge types:
- Explicit cross-references between articles
- Topic mapping (fonon templates ↔ statutory articles)
- Numeric adjacency (article_num ± 1)

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/hasan2salame/Smart-lawyer-assistant.git
cd Smart-lawyer-assistant

# Copy and fill in your API keys
cp .env.example .env

docker compose up
```

Open: **http://localhost:8000**

### Option 2 — Python directly

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
python main.py
# Opens browser automatically at http://localhost:8000
```

---

## First-Time Setup (run once)

If you have raw source files (txt + pdf):

```bash
# 1. Extract structured JSON from raw sources
python scripts/extract_laws.py
python scripts/extract_osoul.py
python scripts/extract_fonon.py   # requires Tesseract + Arabic language pack

# 2. Upload to Qdrant
python scripts/upload_laws.py
python scripts/upload_osoul.py
python scripts/upload_fonon.py

# 3. Build the knowledge graph
python scripts/build_graph.py
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
COHERE_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
GROQ_API_KEY=...
```

---

## Project Structure

```
syrian-legal-assistant/
│
├── config.py                  — Central configuration (all constants)
├── main.py                    — FastAPI server + SSE streaming
├── pipeline.py                — ask() orchestrator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/                   — Source text + PDF files
│   └── processed/             — Extracted JSON + graph.json
│
├── retrieval/
│   ├── __init__.py            — HybridRetriever (main orchestrator)
│   ├── dense.py               — Cohere Embeddings + Qdrant
│   ├── bm25.py                — BM25Okapi sparse search
│   ├── rrf.py                 — Reciprocal Rank Fusion
│   ├── reranker.py            — Cohere cross-encoder reranker
│   ├── adaptive_k.py          — Gap-based dynamic Top-K
│   ├── query_processor.py     — Arabic normalisation + MSA rewriting
│   └── graph/
│       ├── builder.py         — Builds graph.json from Qdrant
│       └── retriever.py       — BFS neighbour expansion
│
├── nlp/
│   ├── classifier.py          — Intent classifier (rules + LLM waterfall)
│   └── llm.py                 — LLM response layer (4 handlers)
│
├── scripts/
│   ├── extract_laws.py        — Personal Status Law TXT → JSON
│   ├── extract_osoul.py       — Civil Procedure Law TXT → JSON
│   ├── extract_fonon.py       — Legal templates PDF → JSON (OCR)
│   ├── upload_laws.py         — Upload laws to Qdrant
│   ├── upload_osoul.py        — Upload osoul to Qdrant
│   ├── upload_fonon.py        — Upload fonon to Qdrant
│   └── build_graph.py         — Build and save knowledge graph
│
├── evaluation/
│   ├── eval_intent.py         — Intent classifier evaluation (40 cases)
│   ├── eval_retrieval.py      — Retrieval evaluation (Recall@K, ablation)
│   └── eval_llm_judge.py      — LLM-as-a-Judge response quality
│
├── tests/
│   ├── test_rrf.py            — rrf_merge() unit tests  (no API needed)
│   ├── test_bm25.py           — BM25 tokeniser + scorer (no API needed)
│   ├── test_graph.py          — GraphRetriever unit tests (no API needed)
│   ├── test_adaptive_k.py     — adaptive_filter() unit tests (no API needed)
│   └── test_dense.py          — DenseSearcher integration tests (live API)
│
└── frontend/
    └── index.html             — Single-page chat interface
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chat interface |
| `GET` | `/health` | Server status + session count |
| `POST` | `/session/new` | Create new session |
| `GET` | `/sessions` | List recent sessions |
| `GET` | `/session/{id}/history` | Full conversation history |
| `DELETE` | `/sessions/all` | Delete ALL sessions and history |
| `DELETE` | `/session/{id}` | Delete a single session |
| `POST` | `/ask` | Send query (supports `stream: true`) |

### POST /ask

```json
{
  "query": "What are the conditions for child custody?",
  "session_id": "optional-uuid",
  "stream": true
}
```

Streaming response (SSE):
```
data: {"type": "meta", "session_id": "...", "intents": ["LEGAL_Q"]}
data: {"type": "token", "token": "According"}
data: {"type": "token", "token": " to..."}
data: {"type": "done", "duration_ms": 1240, "results": [...]}
```

---

## Tech Stack

| Technology | Role |
|-----------|------|
| FastAPI | REST API + Server-Sent Events |
| Qdrant Cloud | Vector database |
| Cohere | Embeddings (`embed-multilingual-v3.0`) + Rerank (`rerank-multilingual-v3.0`) |
| Groq | LLM inference (LLaMA-3.1-8b + LLaMA-3.3-70b) |
| LlamaIndex | Dense vector search bridge |
| BM25Okapi | Sparse retrieval (built from scratch) |
| SQLite | Session persistence |
| Docker | Containerised deployment |

---

## Running Tests

```bash
# Unit tests — no API keys required
python tests/test_rrf.py
python tests/test_bm25.py
python tests/test_graph.py
python tests/test_adaptive_k.py

# Integration tests — requires live Qdrant + Cohere
python tests/test_dense.py

# Evaluation suites — requires all APIs
python evaluation/eval_intent.py
python evaluation/eval_retrieval.py
python evaluation/eval_llm_judge.py
```

---

## License

MIT License — see [LICENSE](LICENSE)
