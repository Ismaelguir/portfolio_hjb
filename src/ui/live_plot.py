# src/ui/live_plot.py
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt

from storage.db import connect
from ui.snapshot import save_snapshot
from cli.commands import cmd_run_daily  # on réutilise la commande existante


def _get_active_symbol(override: str | None = None) -> str:
    if override:
        return override
    conn = connect()
    row = conn.execute("SELECT symbol FROM portfolio_state LIMIT 1;").fetchone()
    if row is None:
        raise RuntimeError("Aucun portfolio_state. Fais init_auto d'abord.")
    return row["symbol"]


def _load_equity(symbol: str) -> pd.DataFrame:
    conn = connect()
    q = """
    SELECT ts_utc, wealth
    FROM equity_curve
    WHERE symbol = ?
    ORDER BY ts_utc ASC
    """
    df = pd.read_sql_query(q, conn, params=(symbol,))
    if len(df) == 0:
        return df
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _load_last_rebalance(symbol: str):
    conn = connect()
    q = """
    SELECT ts_utc, pi, mu_annual, sigma_annual
    FROM rebalances
    WHERE symbol = ?
    ORDER BY ts_utc DESC
    LIMIT 1
    """
    return conn.execute(q, (symbol,)).fetchone()


def run(symbol: str | None = None) -> None:
    symbol = _get_active_symbol(symbol)

    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title(f"portfolio_hjb — {symbol}")
    except Exception:
        pass

    status = fig.text(0.02, 0.98, "", va="top")

    def redraw():
        ax.clear()
        df = _load_equity(symbol)

        if len(df) == 0:
            ax.set_title(f"Equity curve — {symbol} (empty)")
            ax.set_xlabel("Date (UTC)")
            ax.set_ylabel("Wealth")
            status.set_text("Aucune donnée equity_curve.")
            fig.canvas.draw_idle()
            return

        ax.plot(df["ts"], df["wealth"], marker="o")
        ax.set_title(f"Equity curve — {symbol}")
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Wealth")

        last_w = float(df["wealth"].iloc[-1])
        last_t = str(df["ts_utc"].iloc[-1])

        rb = _load_last_rebalance(symbol)
        if rb is None:
            status.set_text(f"Dernier point: {last_t} | wealth={last_w:.6f} | (pas encore de rebalance)")
        else:
            status.set_text(
                f"Dernier point: {last_t} | wealth={last_w:.6f} | "
                f"pi={float(rb['pi']):.6f} | mu={float(rb['mu_annual']):.6f} | sigma={float(rb['sigma_annual']):.6f}"
            )

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "u":
            # Arguments par défaut: alignés sur tes defaults run_daily
            args = SimpleNamespace(
                outputsize="compact",
                window=60,
                annual_days=252,
                nx=400,
                x_min_factor=0.1,
                x_max_factor=3.0,
                xmin_floor=1e-3,
                bc="NEUMANN",
                max_steps=None,
                dry_run=False,
            )
            cmd_run_daily(args)
            redraw()

        elif event.key == "s":
            paths = save_snapshot(symbol)
            status.set_text(f"Snapshot exporté dans outputs/: {paths.png.name}")
            fig.canvas.draw_idle()

        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()
