# src/ui/snapshot.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from storage.db import connect


@dataclass(frozen=True)
class SnapshotPaths:
    equity_csv: Path
    rebalances_csv: Path
    png: Path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_equity_curve(symbol: str, limit: int | None = None) -> pd.DataFrame:
    conn = connect()
    q = """
    SELECT ts_utc, wealth, price, shares_risky, cash_risk_free, note
    FROM equity_curve
    WHERE symbol = ?
    ORDER BY ts_utc ASC
    """
    df = pd.read_sql_query(q, conn, params=(symbol,))
    if limit is not None:
        df = df.tail(int(limit)).reset_index(drop=True)
    return df


def load_rebalances(symbol: str, limit: int | None = None) -> pd.DataFrame:
    conn = connect()
    q = """
    SELECT ts_utc, price, pi, mu_annual, sigma_annual, wealth_before, wealth_after, note
    FROM rebalances
    WHERE symbol = ?
    ORDER BY ts_utc ASC
    """
    df = pd.read_sql_query(q, conn, params=(symbol,))
    if limit is not None:
        df = df.tail(int(limit)).reset_index(drop=True)
    return df


def save_snapshot(symbol: str, out_dir: str = "outputs", limit: int | None = None) -> SnapshotPaths:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()

    eq = load_equity_curve(symbol, limit=limit)
    rb = load_rebalances(symbol, limit=limit)

    equity_csv = out / f"equity_curve_{symbol}_{stamp}.csv"
    rebalances_csv = out / f"rebalances_{symbol}_{stamp}.csv"
    png = out / f"equity_curve_{symbol}_{stamp}.png"

    eq.to_csv(equity_csv, index=False)
    rb.to_csv(rebalances_csv, index=False)

    if len(eq) > 0:
        df = eq.copy()
        df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)

        plt.figure()
        plt.plot(df["ts"], df["wealth"], marker="o")
        plt.title(f"Equity curve — {symbol}")
        plt.xlabel("Date (UTC)")
        plt.ylabel("Wealth")
        plt.tight_layout()
        plt.savefig(png, dpi=160)
        plt.close()
    else:
        # crée quand même un png vide pour cohérence
        plt.figure()
        plt.title(f"Equity curve — {symbol} (empty)")
        plt.tight_layout()
        plt.savefig(png, dpi=160)
        plt.close()

    return SnapshotPaths(equity_csv=equity_csv, rebalances_csv=rebalances_csv, png=png)
