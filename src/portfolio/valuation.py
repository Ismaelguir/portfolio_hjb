# src/portfolio/valuation.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0  # ACT/365


def parse_iso(ts_utc: str) -> datetime:
    # ts_utc est ISO8601 avec offset, ex: "2025-12-25T20:48:22+00:00"
    return datetime.fromisoformat(ts_utc)


def dt_years(ts_from_utc: str, ts_to_utc: str) -> float:
    t0 = parse_iso(ts_from_utc)
    t1 = parse_iso(ts_to_utc)
    dt_seconds = (t1 - t0).total_seconds()
    if dt_seconds < 0:
        return 0.0
    return dt_seconds / SECONDS_PER_YEAR


def grow_cash(cash: float, r_annual: float, delta_years: float) -> float:
    # B <- B * exp(r dt)
    import math
    return float(cash) * math.exp(float(r_annual) * float(delta_years))


def wealth(shares_risky: float, price: float, cash: float) -> float:
    return float(shares_risky) * float(price) + float(cash)
