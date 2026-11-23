"""SQLite review queue. One row per processed document; corrections are also written out as
labelled examples (image + target JSON) that the fine-tuning notebook can train on."""

import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,          -- accepted | review | corrected
    image BLOB NOT NULL,           -- PNG
    prediction TEXT NOT NULL,      -- pipeline record
    assessment TEXT NOT NULL,      -- checks, field confidence, score
    correction TEXT,
    reviewed_at REAL
)"""


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def add(conn: sqlite3.Connection, png: bytes, prediction: dict, assessment: dict) -> str:
    doc_id = uuid.uuid4().hex[:12]
    status = "accepted" if assessment["accept"] else "review"
    conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                 (doc_id, time.time(), status, png, json.dumps(prediction), json.dumps(assessment)))
    conn.commit()
    return doc_id


def get(conn: sqlite3.Connection, doc_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    for k in ("prediction", "assessment", "correction"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d


def pending(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT id FROM documents WHERE status = 'review' ORDER BY created_at")
    return [r["id"] for r in rows]


def save_correction(conn: sqlite3.Connection, doc_id: str, receipt: dict,
                    corrections_dir: str) -> Path:
    """Mark the document corrected and write <dir>/<id>.png + <id>.json, the same shape as the
    CORD-derived training targets (values as printed)."""
    doc = get(conn, doc_id)
    if doc is None:
        raise KeyError(doc_id)
    out = Path(corrections_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{doc_id}.png").write_bytes(doc["image"])
    (out / f"{doc_id}.json").write_text(json.dumps(
        {"image": f"{doc_id}.png", "target": receipt}, ensure_ascii=False, indent=1))
    conn.execute("UPDATE documents SET status = 'corrected', correction = ?, reviewed_at = ? "
                 "WHERE id = ?", (json.dumps(receipt), time.time(), doc_id))
    conn.commit()
    return out / f"{doc_id}.json"
