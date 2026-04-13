"""
main.py
FastAPI — المساعد القانوني السوري

الميزات:
  ✓ يفتح المتصفح تلقائياً عند التشغيل
  ✓ يخدم الواجهة مباشرة (لا حاجة لفتح index.html يدوياً)
  ✓ SQLite لحفظ المحادثات
  ✓ Streaming حقيقي من Groq
  ✓ تعدد نوافذ مستقلة
"""

import uuid
import time
import json
import sqlite3
import threading
import webbrowser
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from pipeline import ask
from nlp.llm import process_stream
from nlp.classifier import classify
from config import DB_PATH, FRONTEND_DIR, HOST, PORT

# ── قاعدة البيانات ─────────────────────────────────────────────────────

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title      TEXT DEFAULT 'محادثة جديدة',
                created_at REAL,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                session_id    TEXT PRIMARY KEY,
                last_template TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.commit()


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _get_history(session_id: str, limit: int = 20) -> list:
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def _get_last_template(session_id: str) -> dict | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT last_template FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    if row and row["last_template"]:
        try:
            return json.loads(row["last_template"])
        except Exception:
            return None
    return None


def _save_message(session_id: str, role: str, content: str):
    now = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, now)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )
        row = conn.execute(
            "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row and row["title"] == "محادثة جديدة" and role == "user":
            title = content[:40] + ("…" if len(content) > 40 else "")
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                (title, session_id)
            )


def _save_template(session_id: str, template: dict | None):
    value = json.dumps(template, ensure_ascii=False) if template else None
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO session_state (session_id, last_template) VALUES (?,?)",
            (session_id, value)
        )


def _ensure_session(session_id: str):
    now = time.time()
    with _db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (session_id, "محادثة جديدة", now, now)
            )


# ── FastAPI ─────────────────────────────────────────────────────────────

app = FastAPI(title="المساعد القانوني السوري", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_init_db()


# ── خدمة الواجهة ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    """يُرجع الواجهة مباشرة"""
    return FileResponse(FRONTEND_DIR / "index.html")


# ── Models ────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query:      str           = Field(..., min_length=1)
    session_id: Optional[str] = None
    stream:     bool          = False


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with _db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    return {"status": "ok", "version": "1.0.0", "sessions": count}


@app.post("/session/new")
def new_session():
    sid = str(uuid.uuid4())
    _ensure_session(sid)
    return {"session_id": sid}


@app.get("/sessions")
def list_sessions(limit: int = 20):
    with _db() as conn:
        rows = conn.execute(
            "SELECT session_id, title, updated_at FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"session_id": r["session_id"], "title": r["title"], "updated_at": r["updated_at"]}
            for r in rows]


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    return {"session_id": session_id, "messages": _get_history(session_id, limit=100)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    with _db() as conn:
        conn.execute("DELETE FROM messages     WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_state WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions     WHERE session_id = ?", (session_id,))
    return {"deleted": True}


@app.post("/ask")
def ask_endpoint(body: AskRequest):
    start = time.perf_counter()
    sid   = body.session_id or str(uuid.uuid4())
    _ensure_session(sid)

    history       = _get_history(sid)
    last_template = _get_last_template(sid)

    _save_message(sid, "user", body.query)

    if body.stream:
        def _stream():
            classification = classify(body.query)
            intents        = classification["intents"]

            yield "data: " + json.dumps({
                "type": "meta", "session_id": sid, "intents": intents,
            }, ensure_ascii=False) + "\n\n"

            rag_meta, gen = process_stream(
                query=body.query, intents=intents,
                history=history, last_template=last_template,
            )

            if gen is not None:
                full = ""
                for token in gen:
                    full += token
                    yield "data: " + json.dumps({"type": "token", "token": token},
                                                 ensure_ascii=False) + "\n\n"
                _save_message(sid, "assistant", full)
                duration = round((time.perf_counter() - start) * 1000, 1)
                yield "data: " + json.dumps({
                    "type": "done", "duration_ms": duration,
                    "results": [{"intent": "LEGAL_Q", "message": full,
                                 "articles": rag_meta.get("articles", []),
                                 "template": None, "attachments": None}],
                }, ensure_ascii=False) + "\n\n"
            else:
                result = ask(query=body.query, history=history, last_template=last_template)
                _save_message(sid, "assistant", result["message"])
                if result.get("_template"):
                    _save_template(sid, result["_template"])

                for sentence in result["message"].replace("\n", "\n ").split("\n"):
                    if sentence.strip():
                        yield "data: " + json.dumps({"type": "token", "token": sentence + "\n"},
                                                     ensure_ascii=False) + "\n\n"

                duration = round((time.perf_counter() - start) * 1000, 1)
                yield "data: " + json.dumps({
                    "type": "done", "duration_ms": duration,
                    "results": result.get("results", []),
                }, ensure_ascii=False) + "\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Session-Id": sid},
        )

    try:
        result = ask(query=body.query, history=history, last_template=last_template)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _save_message(sid, "assistant", result["message"])
    if result.get("_template"):
        _save_template(sid, result["_template"])

    return {
        "session_id":  sid,
        "intents":     result["intents"],
        "message":     result["message"],
        "results":     result.get("results", []),
        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
    }


# ── تشغيل مباشر ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # فتح المتصفح تلقائياً بعد ثانية واحدة من بدء السيرفر
    def _open_browser():
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Timer(1.5, _open_browser).start()
    print(f"\n✅ المساعد القانوني يعمل على: http://localhost:{PORT}\n")

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
