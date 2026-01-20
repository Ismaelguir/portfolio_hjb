# src/storage/repo.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Optional
import pandas as pd


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

def upsert_equity_point(
    conn: sqlite3.Connection,
    symbol: str,
    ts_utc: str,
    wealth: float,
    price: float,
    shares_risky: float,
    cash_risk_free: float,
    note: str | None = None,
) -> None:
    """
    Enregistre un point de courbe de richesse.
    Upsert sur (symbol, ts_utc) : si le point existe, on le met à jour.
    """
    conn.execute(
        """
        INSERT INTO equity_curve(symbol, ts_utc, wealth, price, shares_risky, cash_risk_free, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts_utc) DO UPDATE SET
          wealth=excluded.wealth,
          price=excluded.price,
          shares_risky=excluded.shares_risky,
          cash_risk_free=excluded.cash_risk_free,
          note=excluded.note
        """,
        (symbol, ts_utc, float(wealth), float(price), float(shares_risky), float(cash_risk_free), note),
    )
    conn.commit()


def get_equity_series(
    conn: sqlite3.Connection,
    symbol: str,
    start_ts_utc: str | None = None,
    end_ts_utc: str | None = None,
) -> pd.DataFrame:
    """
    Renvoie la série de richesse (ts_utc, wealth, price, shares_risky, cash_risk_free, note)
    triée par ts_utc croissant.
    """
    q = """
      SELECT ts_utc, wealth, price, shares_risky, cash_risk_free, note
      FROM equity_curve
      WHERE symbol=?
    """
    params: list[object] = [symbol]
    if start_ts_utc is not None:
        q += " AND ts_utc >= ?"
        params.append(start_ts_utc)
    if end_ts_utc is not None:
        q += " AND ts_utc < ?"
        params.append(end_ts_utc)
    q += " ORDER BY ts_utc ASC"

    return pd.read_sql_query(q, conn, params=params)


def get_latest_equity_point(conn: sqlite3.Connection, symbol: str) -> Optional[tuple[str, float]]:
    """
    Dernier point de richesse : (ts_utc, wealth) ou None si vide.
    """
    row = conn.execute(
        """
        SELECT ts_utc, wealth
        FROM equity_curve
        WHERE symbol=?
        ORDER BY ts_utc DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if row is None:
        return None
    return (row["ts_utc"], float(row["wealth"]))

def insert_rebalance(
    conn,
    symbol: str,
    ts_utc: str,
    price: float,
    wealth_before: float,
    wealth_after: float,
    shares_risky: float,
    cash_risk_free: float,
    pi: float,
    mu_annual: float | None,
    sigma_annual: float | None,
    window: int | None,
    annual_days: int | None,
    note: str = "",
    sigma_method: str | None = None,
    ewma_lambda: float | None = None,
    gbm_correction: bool | None = None,
) -> None:
    gbm_int = None if gbm_correction is None else (1 if gbm_correction else 0)

    conn.execute(
        """
        INSERT INTO rebalances(
          symbol, ts_utc, price, wealth_before, wealth_after,
          shares_risky, cash_risk_free, pi,
          mu_annual, sigma_annual, sigma_method, ewma_lambda, gbm_correction,
          window, annual_days, note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            ts_utc,
            float(price),
            float(wealth_before),
            float(wealth_after),
            float(shares_risky),
            float(cash_risk_free),
            float(pi),
            None if mu_annual is None else float(mu_annual),
            None if sigma_annual is None else float(sigma_annual),
            sigma_method,
            ewma_lambda,
            gbm_int,
            window,
            annual_days,
            note,
        ),
    )
    conn.commit()



from dataclasses import dataclass
import sqlite3


@dataclass(frozen=True)
class RunConfig:
    mu_window: int
    sigma_window: int
    annual_days: int
    sigma_method: str        # "std" or "ewma"
    ewma_lambda: float | None
    gbm_correction: bool
    nx: int
    bc: str
    x_min_factor: float
    x_max_factor: float
    xmin_floor: float



def get_run_config(conn: sqlite3.Connection) -> RunConfig:
    row = conn.execute(
        """
        SELECT
          mu_window, sigma_window, annual_days,
          sigma_method, ewma_lambda, gbm_correction,
          nx, bc, x_min_factor, x_max_factor, xmin_floor
        FROM run_config
        WHERE id=1
        """
    ).fetchone()

    return RunConfig(
        mu_window=int(row["mu_window"]),
        sigma_window=int(row["sigma_window"]),
        annual_days=int(row["annual_days"]),
        sigma_method=str(row["sigma_method"]),
        ewma_lambda=(None if row["ewma_lambda"] is None else float(row["ewma_lambda"])),
        gbm_correction=bool(int(row["gbm_correction"])),
        nx=int(row["nx"]),
        bc=str(row["bc"]),
        x_min_factor=float(row["x_min_factor"]),
        x_max_factor=float(row["x_max_factor"]),
        xmin_floor=float(row["xmin_floor"]),
    )



def set_run_config(conn: sqlite3.Connection, cfg: RunConfig) -> None:
    conn.execute(
        """
        UPDATE run_config
        SET
          mu_window=?,
          sigma_window=?,
          annual_days=?,
          sigma_method=?,
          ewma_lambda=?,
          gbm_correction=?,
          nx=?,
          bc=?,
          x_min_factor=?,
          x_max_factor=?,
          xmin_floor=?
        WHERE id=1
        """,
        (
            int(cfg.mu_window),
            int(cfg.sigma_window),
            int(cfg.annual_days),
            str(cfg.sigma_method),
            (None if cfg.sigma_method == "std" else cfg.ewma_lambda),
            1 if cfg.gbm_correction else 0,
            int(cfg.nx),
            str(cfg.bc),
            float(cfg.x_min_factor),
            float(cfg.x_max_factor),
            float(cfg.xmin_floor),
        ),
    )
    conn.commit()
