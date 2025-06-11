"""
SQLite helper for Personal Garden Beds (Collections)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Dict, Any

DB_PATH = Path(__file__).parent / "garden.sqlite3"

# Ensure parent directory exists (should already)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    # Enable foreign keys for cascade delete
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not yet exist"""
    with _get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS collection_items (
                collection_id INTEGER NOT NULL,
                prompt_id TEXT NOT NULL,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (collection_id, prompt_id),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


# --- CRUD helpers ---------------------------------------------------------


def create_collection(user_id: str, name: str) -> int:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO collections (user_id, name) VALUES (?, ?)",
            (user_id, name),
        )
        conn.commit()
        return cur.lastrowid


def list_collections(user_id: str) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, created_at FROM collections WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = cur.fetchall()
        return [dict(id=r[0], name=r[1], created_at=r[2]) for r in rows]


def add_item(collection_id: int, prompt_id: str) -> None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO collection_items (collection_id, prompt_id) VALUES (?, ?)",
            (collection_id, prompt_id),
        )
        conn.commit()


def remove_item(collection_id: int, prompt_id: str) -> None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM collection_items WHERE collection_id = ? AND prompt_id = ?",
            (collection_id, prompt_id),
        )
        conn.commit()


def get_collection_items(collection_id: int) -> List[str]:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT prompt_id FROM collection_items WHERE collection_id = ?",
            (collection_id,),
        )
        return [row[0] for row in cur.fetchall()]


# Initialize DB when module imported
init_db() 