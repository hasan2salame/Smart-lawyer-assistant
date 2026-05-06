"""
config.py
Central configuration — all constants in one place.
Sensitive values are read from .env only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
GRAPH_FILE     = DATA_PROCESSED / "graph.json"
DB_PATH        = BASE_DIR / "sessions.db"
FRONTEND_DIR   = BASE_DIR / "frontend"

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

# ── API Keys (loaded from .env) ───────────────────────────────────────────
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")

# ── Confidence Threshold ─────────────────────────────────────────────────
# Kept for check_confidence() utility and tests only.
# NOT used to reject queries in adaptive_filter() — Cohere rerank scores are
# relative (not absolute probabilities), so a fixed threshold on absolute
# scores causes false negatives for valid legal questions in this corpus.
# Out-of-scope routing is handled by:
#   (1) intent classifier  — blocks non-legal queries before retrieval
#   (2) empty retrieval list — adaptive_filter returns [] only when no articles found
CONFIDENCE_THRESHOLD = 0.10

# ── Adaptive K Bounds ─────────────────────────────────────────────────────
ADAPTIVE_K_MIN = 1
ADAPTIVE_K_MAX = 10
