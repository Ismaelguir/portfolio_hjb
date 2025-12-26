# src/storage/repo.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PortfolioState:
    symbol: str
    risk_free_rate: float  # r annuel
    gamma: float           # CRRA
    horizon_years: float   # T
    pi_min: float
    pi_max: float

    shares_risky: float    # n
    cash_risk_free: float  # B

    last_ts_utc: str
    last_price: float
    last_wealth: float


def upsert_state(conn: sqlite3.Connection, state: PortfolioState) -> None:
    """Insère ou remplace l'unique ligne id=1."""
    conn.execute(
        """
        INSERT INTO portfolio_state (
          id, symbol, risk_free_rate, gamma, horizon_years, pi_min, pi_max,
          shares_risky, cash_risk_free, last_ts_utc, last_price, last_wealth
        )
        VALUES (
          1, :symbol, :risk_free_rate, :gamma, :horizon_years, :pi_min, :pi_max,
          :shares_risky, :cash_risk_free, :last_ts_utc, :last_price, :last_wealth
        )
        ON CONFLICT(id) DO UPDATE SET
          symbol=excluded.symbol,
          risk_free_rate=excluded.risk_free_rate,
          gamma=excluded.gamma,
          horizon_years=excluded.horizon_years,
          pi_min=excluded.pi_min,
          pi_max=excluded.pi_max,
          shares_risky=excluded.shares_risky,
          cash_risk_free=excluded.cash_risk_free,
          last_ts_utc=excluded.last_ts_utc,
          last_price=excluded.last_price,
          last_wealth=excluded.last_wealth
        """,
        {
            "symbol": state.symbol,
            "risk_free_rate": state.risk_free_rate,
            "gamma": state.gamma,
            "horizon_years": state.horizon_years,
            "pi_min": state.pi_min,
            "pi_max": state.pi_max,
            "shares_risky": state.shares_risky,
            "cash_risk_free": state.cash_risk_free,
            "last_ts_utc": state.last_ts_utc,
            "last_price": state.last_price,
            "last_wealth": state.last_wealth,
        },
    )
    conn.commit()


def load_state(conn: sqlite3.Connection) -> Optional[PortfolioState]:
    row = conn.execute("SELECT * FROM portfolio_state WHERE id=1").fetchone()
    if row is None:
        return None
    return PortfolioState(
        symbol=row["symbol"],
        risk_free_rate=row["risk_free_rate"],
        gamma=row["gamma"],
        horizon_years=row["horizon_years"],
        pi_min=row["pi_min"],
        pi_max=row["pi_max"],
        shares_risky=row["shares_risky"],
        cash_risk_free=row["cash_risk_free"],
        last_ts_utc=row["last_ts_utc"],
        last_price=row["last_price"],
        last_wealth=row["last_wealth"],
    )


def log_event(conn: sqlite3.Connection, event_type: str, payload: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO events (ts_utc, event_type, payload) VALUES (?, ?, ?)",
        (utc_now_iso(), event_type, json.dumps(payload) if payload is not None else None),
    )
    conn.commit()
