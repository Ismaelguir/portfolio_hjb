# src/ui/queries.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import pandas as pd

from storage.db import connect


def _date_bounds_utc(d: date) -> tuple[str, str]:
    # Inclusif sur la journée, format cohérent avec tes ts_utc ("...+00:00")
    start = f"{d.isoformat()}T00:00:00+00:00"
    end = f"{d.isoformat()}T23:59:59+00:00"
    return start, end


def list_symbols() -> list[str]:
    conn = connect()
    rows = conn.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol;").fetchall()
    return [r["symbol"] for r in rows]


def list_granularities(symbol: str | None = None) -> list[str]:
    conn = connect()
    if symbol:
        rows = conn.execute(
            "SELECT DISTINCT granularity FROM prices WHERE symbol=? ORDER BY granularity;",
            (symbol,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT DISTINCT granularity FROM prices ORDER BY granularity;").fetchall()
    return [r["granularity"] for r in rows]


def get_prices(
    symbol: str,
    granularity: str = "DAILY",
    start: date | None = None,
    end: date | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    conn = connect()
    where = ["symbol = ?", "granularity = ?"]
    params: list[Any] = [symbol, granularity]

    if start:
        s, _ = _date_bounds_utc(start)
        where.append("ts_utc >= ?")
        params.append(s)
    if end:
        _, e = _date_bounds_utc(end)
        where.append("ts_utc <= ?")
        params.append(e)

    q = f"""
    SELECT ts_utc, price, source
    FROM prices
    WHERE {' AND '.join(where)}
    ORDER BY ts_utc DESC
    LIMIT ?
    """
    params.append(int(limit))
    df = pd.read_sql_query(q, conn, params=tuple(params))
    return df


def get_equity_curve(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    conn = connect()
    where = ["symbol = ?"]
    params: list[Any] = [symbol]

    if start:
        s, _ = _date_bounds_utc(start)
        where.append("ts_utc >= ?")
        params.append(s)
    if end:
        _, e = _date_bounds_utc(end)
        where.append("ts_utc <= ?")
        params.append(e)

    q = f"""
    SELECT ts_utc, wealth, price, shares_risky, cash_risk_free, note
    FROM equity_curve
    WHERE {' AND '.join(where)}
    ORDER BY ts_utc DESC
    LIMIT ?
    """
    params.append(int(limit))
    df = pd.read_sql_query(q, conn, params=tuple(params))
    return df


def get_rebalances(symbol: str, limit: int = 200) -> pd.DataFrame:
    conn = connect()
    q = """
    SELECT
      ts_utc,
      price,
      pi,
      mu_annual,
      sigma_annual,
      sigma_method,
      ewma_lambda,
      gbm_correction,
      wealth_before,
      wealth_after,
      note
    FROM rebalances
    WHERE symbol = ?
    ORDER BY ts_utc DESC
    LIMIT ?
    """
    return pd.read_sql_query(q, conn, params=(symbol, int(limit)))



def get_events(limit: int = 300) -> pd.DataFrame:
    conn = connect()
    q = """
    SELECT ts_utc, event_type, payload
    FROM events
    ORDER BY ts_utc DESC
    LIMIT ?
    """
    return pd.read_sql_query(q, conn, params=(int(limit),))


def get_state() -> pd.DataFrame:
    conn = connect()
    q = """
    SELECT symbol, risk_free_rate, gamma, horizon_years, pi_min, pi_max,
           shares_risky, cash_risk_free, last_ts_utc, last_price, last_wealth
    FROM portfolio_state
    LIMIT 1
    """
    return pd.read_sql_query(q, conn)
