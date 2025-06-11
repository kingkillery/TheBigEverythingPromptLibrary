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

        # Usage stats table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_stats (
                prompt_id TEXT PRIMARY KEY,
                views INTEGER DEFAULT 0,
                grafts INTEGER DEFAULT 0,
                last_viewed TEXT,
                last_grafted TEXT
            )
            """
        )
        conn.commit()

        # Prompt chains (vines) table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                prompts TEXT NOT NULL,  -- JSON array of prompt objects
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
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


# ------------------ Usage stats helpers -----------------------------------


def record_view(prompt_id: str):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO usage_stats (prompt_id, views, last_viewed) VALUES (?, 1, CURRENT_TIMESTAMP) "
            "ON CONFLICT(prompt_id) DO UPDATE SET views = views + 1, last_viewed = CURRENT_TIMESTAMP",
            (prompt_id,),
        )
        conn.commit()


def record_graft(prompt_id: str):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO usage_stats (prompt_id, grafts, last_grafted) VALUES (?, 1, CURRENT_TIMESTAMP) "
            "ON CONFLICT(prompt_id) DO UPDATE SET grafts = grafts + 1, last_grafted = CURRENT_TIMESTAMP",
            (prompt_id,),
        )
        conn.commit()


def get_popular(limit: int = 10):
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT prompt_id, views, grafts FROM usage_stats ORDER BY views DESC LIMIT ?",
            (limit,),
        )
        return [dict(prompt_id=r[0], views=r[1], grafts=r[2]) for r in cur.fetchall()]


def get_trending(limit: int = 10):
    """Get prompts trending in the last 7 days"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT prompt_id, views, grafts, last_viewed, last_grafted,
                   (views + grafts * 2) as trending_score
            FROM usage_stats 
            WHERE last_viewed >= datetime('now', '-7 days') 
               OR last_grafted >= datetime('now', '-7 days')
            ORDER BY trending_score DESC, last_viewed DESC 
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(
            prompt_id=r[0], 
            views=r[1], 
            grafts=r[2], 
            last_viewed=r[3],
            last_grafted=r[4],
            trending_score=r[5]
        ) for r in cur.fetchall()]


def get_most_grafted(limit: int = 10):
    """Get most grafted (remixed) prompts"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT prompt_id, views, grafts FROM usage_stats WHERE grafts > 0 ORDER BY grafts DESC LIMIT ?",
            (limit,),
        )
        return [dict(prompt_id=r[0], views=r[1], grafts=r[2]) for r in cur.fetchall()]


def get_new_sprouts(limit: int = 10):
    """Get recently viewed prompts (new discoveries)"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT prompt_id, views, grafts, last_viewed
            FROM usage_stats 
            WHERE last_viewed >= datetime('now', '-3 days')
            ORDER BY last_viewed DESC 
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(
            prompt_id=r[0], 
            views=r[1], 
            grafts=r[2],
            last_viewed=r[3]
        ) for r in cur.fetchall()]


def rename_collection(collection_id: int, new_name: str) -> None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE collections SET name = ? WHERE id = ?",
            (new_name, collection_id),
        )
        conn.commit()


def delete_collection(collection_id: int) -> None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM collections WHERE id = ?",
            (collection_id,),
        )
        conn.commit()


# -------------------- Prompt Chains (Vines) Management -------------------

def create_chain(user_id: str, name: str, description: str, prompts: str) -> int:
    """Create a new prompt chain"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prompt_chains (user_id, name, description, prompts) VALUES (?, ?, ?, ?)",
            (user_id, name, description, prompts),
        )
        conn.commit()
        return cur.lastrowid


def list_chains(user_id: str) -> List[Dict[str, Any]]:
    """List all chains for a user"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, description, prompts, created_at FROM prompt_chains WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = cur.fetchall()
        return [dict(
            id=r[0], 
            name=r[1], 
            description=r[2], 
            prompts=r[3], 
            created_at=r[4]
        ) for r in rows]


def get_chain(chain_id: int, user_id: str) -> Dict[str, Any]:
    """Get a specific chain"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, description, prompts, created_at FROM prompt_chains WHERE id = ? AND user_id = ?",
            (chain_id, user_id),
        )
        row = cur.fetchone()
        if row:
            return dict(
                id=row[0], 
                name=row[1], 
                description=row[2], 
                prompts=row[3], 
                created_at=row[4]
            )
        return None


def update_chain(chain_id: int, user_id: str, name: str = None, description: str = None, prompts: str = None):
    """Update a chain"""
    with _get_conn() as conn:
        cur = conn.cursor()
        if name is not None:
            cur.execute(
                "UPDATE prompt_chains SET name = ? WHERE id = ? AND user_id = ?",
                (name, chain_id, user_id),
            )
        if description is not None:
            cur.execute(
                "UPDATE prompt_chains SET description = ? WHERE id = ? AND user_id = ?",
                (description, chain_id, user_id),
            )
        if prompts is not None:
            cur.execute(
                "UPDATE prompt_chains SET prompts = ? WHERE id = ? AND user_id = ?",
                (prompts, chain_id, user_id),
            )
        conn.commit()


def delete_chain(chain_id: int, user_id: str) -> None:
    """Delete a chain"""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM prompt_chains WHERE id = ? AND user_id = ?",
            (chain_id, user_id),
        )
        conn.commit()


# Initialize DB when module imported
init_db() 