# src/market/price_cache.py
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional
import pandas as pd


def insert_prices(
    conn: sqlite3.Connection,
    symbol: str,
    granularity: str,
    rows: Iterable[tuple[str, float]],
    source: str = "CSV",
) -> int:
    """
    Insère des points de prix (ts_utc, price).
    Si un point existe déjà, on met à jour le prix (upsert).
    Retourne le nombre de lignes écrites (SQLite ne donne pas toujours un count fiable sur upsert,
    donc on retourne len(list(rows)) après matérialisation).
    """
    rows = list(rows)
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO prices(symbol, granularity, ts_utc, price, source)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol, granularity, ts_utc) DO UPDATE SET
          price=excluded.price,
          source=excluded.source
        """,
        [(symbol, granularity, ts, float(price), source) for ts, price in rows],
    )
    conn.commit()
    return len(rows)


def get_prices(
    conn: sqlite3.Connection,
    symbol: str,
    granularity: str,
    start_ts_utc: Optional[str] = None,
    end_ts_utc: Optional[str] = None,
) -> pd.DataFrame:
    """
    Renvoie un DataFrame avec colonnes: ts_utc, price (trié croissant).
    start/end sont inclusifs/exclusifs comme suit:
      ts >= start_ts_utc si start fourni
      ts <  end_ts_utc   si end fourni
    """
    query = """
      SELECT ts_utc, price
      FROM prices
      WHERE symbol=? AND granularity=?
    """
    params: list[object] = [symbol, granularity]

    if start_ts_utc is not None:
        query += " AND ts_utc >= ?"
        params.append(start_ts_utc)
    if end_ts_utc is not None:
        query += " AND ts_utc < ?"
        params.append(end_ts_utc)

    query += " ORDER BY ts_utc ASC"

    df = pd.read_sql_query(query, conn, params=params)
    return df


def get_latest_price(
    conn: sqlite3.Connection,
    symbol: str,
    granularity: str,
) -> Optional[tuple[str, float]]:
    """
    Dernier point (ts_utc, price) ou None si vide.
    """
    row = conn.execute(
        """
        SELECT ts_utc, price
        FROM prices
        WHERE symbol=? AND granularity=?
        ORDER BY ts_utc DESC
        LIMIT 1
        """,
        (symbol, granularity),
    ).fetchone()

    if row is None:
        return None
    return (row["ts_utc"], float(row["price"]))
