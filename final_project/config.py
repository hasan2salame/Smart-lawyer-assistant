"""
config.py
الإعدادات المركزية — كل الثوابت في مكان واحد

القيم الحساسة تُقرأ من .env فقط.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── المسارات ──────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_RAW      = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
GRAPH_FILE    = DATA_PROCESSED / "graph.json"
DB_PATH       = BASE_DIR / "sessions.db"
FRONTEND_DIR  = BASE_DIR / "frontend"

# ── Qdrant Collections ────────────────────────────────────────────────────
COL_LAWS  = "legal_laws"
COL_OSOUL = "legal_osoul"
COL_FONON = "legal_fonon"

# ── Cohere ────────────────────────────────────────────────────────────────
COHERE_EMBED_MODEL  = "embed-multilingual-v3.0"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"

# ── Groq Models ───────────────────────────────────────────────────────────
MODEL_FAST  = "llama-3.1-8b-instant"
MODEL_LEGAL = "llama-3.3-70b-versatile"

# ── Retrieval Parameters ─────────────────────────────────────────────────
TOP_K_DENSE = 20
TOP_K_BM25  = 20
TOP_K_RRF   = 15
TOP_K_FINAL = 6
RRF_K       = 60
GRAPH_DECAY = 0.7

# ── Session ───────────────────────────────────────────────────────────────
MAX_HISTORY = 20

# ── Server ────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── API Keys ──────────────────────────────────────────────────────────────
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
