# src/storage/db.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import SCHEMA_SQL


def get_db_path() -> Path:
    return Path("data") / "portfolio.db"


def connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # permet d'accéder aux colonnes par nom
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Crée les tables si elles n'existent pas."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
