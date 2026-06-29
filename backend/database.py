"""nChat - Database layer using SQLite"""
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "nchat.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Conversation',
                model TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                thinking TEXT NOT NULL DEFAULT '',
                sources TEXT NOT NULL DEFAULT '',
                model TEXT DEFAULT '',
                tokens_eval INTEGER DEFAULT 0,
                tokens_prompt INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                message_id TEXT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                est_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_files_message
                ON files(message_id);

            CREATE INDEX IF NOT EXISTS idx_files_conversation
                ON files(conversation_id);
        """)

        # --- Migrations for DBs created before these columns existed ---
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "system_prompt" not in cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''"
            )

        mcols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "thinking" not in mcols:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN thinking TEXT NOT NULL DEFAULT ''"
            )
        if "sources" not in mcols:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT ''"
            )

        # Seed starter personas only if none exist (never overwrites user edits)
        count = conn.execute("SELECT COUNT(*) AS n FROM prompts").fetchone()["n"]
        if count == 0:
            for name, content in DEFAULT_PROMPTS:
                conn.execute(
                    "INSERT INTO prompts (id, name, content, created_at) VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), name, content, now_iso())
                )


DEFAULT_PROMPTS = [
    ("Network Engineer",
     "You are a senior network engineer. Be precise and protocol-accurate "
     "(IOS/NX-OS/Junos). When reviewing configs, call out errors, security "
     "issues, and MTU/BGP/OSPF problems explicitly. Prefer concrete commands "
     "over prose. Say when something is ambiguous rather than guessing."),
    ("Config Reviewer",
     "You review network device configurations for correctness, security, and "
     "consistency. Flag issues by severity. Quote the exact offending lines. "
     "Do not invent configuration that isn't present in the provided file."),
    ("Concise Assistant",
     "Answer directly and concisely. Skip preamble. Use short paragraphs and "
     "code blocks where they help. If you are uncertain, say so."),
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --- Conversation CRUD ---

def create_conversation(model: str = "", title: str = "New Conversation",
                        system_prompt: str = "") -> dict:
    conv_id = str(uuid.uuid4())
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, model, system_prompt, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, title, model, system_prompt, ts, ts)
        )
    return {"id": conv_id, "title": title, "model": model,
            "system_prompt": system_prompt, "created_at": ts, "updated_at": ts}


def list_conversations(limit: int = 50, offset: int = 0) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return dict(row) if row else None


def update_conversation(conv_id: str, **kwargs) -> dict | None:
    allowed = {"title", "model", "system_prompt"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_conversation(conv_id)
    updates["updated_at"] = now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [conv_id]
    with get_db() as conn:
        conn.execute(f"UPDATE conversations SET {set_clause} WHERE id = ?", values)
    return get_conversation(conv_id)


def delete_conversation(conv_id: str) -> bool:
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    return cursor.rowcount > 0


# --- Message CRUD ---

def add_message(conversation_id: str, role: str, content: str,
                model: str = "", tokens_eval: int = 0, tokens_prompt: int = 0,
                duration_ms: int = 0, thinking: str = "", sources: str = "") -> dict:
    msg_id = str(uuid.uuid4())
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO messages
               (id, conversation_id, role, content, thinking, sources, model, tokens_eval, tokens_prompt, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, conversation_id, role, content, thinking, sources, model, tokens_eval, tokens_prompt, duration_ms, ts)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (ts, conversation_id)
        )
    return {
        "id": msg_id, "conversation_id": conversation_id, "role": role,
        "content": content, "thinking": thinking, "sources": sources, "model": model,
        "tokens_eval": tokens_eval, "tokens_prompt": tokens_prompt,
        "duration_ms": duration_ms, "created_at": ts
    }


def get_messages(conversation_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def auto_title(conversation_id: str):
    """Set conversation title from first user message if still default."""
    conv = get_conversation(conversation_id)
    if not conv or conv["title"] != "New Conversation":
        return
    messages = get_messages(conversation_id)
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:80].strip()
            if len(msg["content"]) > 80:
                title += "..."
            update_conversation(conversation_id, title=title)
            break


# --- File CRUD ---

def create_file(filename: str, content: str, size_bytes: int = 0,
                est_tokens: int = 0, conversation_id: str | None = None) -> dict:
    """Store an uploaded clear-text file. Not yet linked to a message —
    that happens at send time via attach_files_to_message()."""
    file_id = str(uuid.uuid4())
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO files
               (id, conversation_id, message_id, filename, content, size_bytes, est_tokens, created_at)
               VALUES (?, ?, NULL, ?, ?, ?, ?, ?)""",
            (file_id, conversation_id, filename, content, size_bytes, est_tokens, ts)
        )
    # Return metadata only — never echo content back on upload.
    return {
        "id": file_id, "filename": filename, "size_bytes": size_bytes,
        "est_tokens": est_tokens, "created_at": ts
    }


def get_file(file_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    return dict(row) if row else None


def attach_files_to_message(message_id: str, file_ids: list[str],
                            conversation_id: str | None = None):
    """Bind uploaded files to the message they were sent with, so history
    carries them forward on later turns."""
    if not file_ids:
        return
    with get_db() as conn:
        for fid in file_ids:
            conn.execute(
                """UPDATE files
                   SET message_id = ?,
                       conversation_id = COALESCE(conversation_id, ?)
                   WHERE id = ?""",
                (message_id, conversation_id, fid)
            )


def get_files_for_message(message_id: str) -> list[dict]:
    """Files attached to a given message, used to rebuild prompt context."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, filename, content, size_bytes, est_tokens
               FROM files WHERE message_id = ? ORDER BY created_at ASC""",
            (message_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_file_meta_for_message(message_id: str) -> list[dict]:
    """Same as above but without content — safe to ship to the frontend
    for rendering file chips."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, filename, size_bytes, est_tokens
               FROM files WHERE message_id = ? ORDER BY created_at ASC""",
            (message_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- Prompt / persona CRUD ---

def list_prompts() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, content, created_at FROM prompts ORDER BY name ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def create_prompt(name: str, content: str) -> dict:
    prompt_id = str(uuid.uuid4())
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO prompts (id, name, content, created_at) VALUES (?, ?, ?, ?)",
            (prompt_id, name, content, ts)
        )
    return {"id": prompt_id, "name": name, "content": content, "created_at": ts}


def delete_prompt(prompt_id: str) -> bool:
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    return cursor.rowcount > 0