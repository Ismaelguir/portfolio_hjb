# src/main.py

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()



import argparse

from storage.db import connect, init_db
from storage.repo import PortfolioState, upsert_state, load_state, log_event, utc_now_iso
from storage.repo import upsert_equity_point, get_equity_series, get_latest_equity_point
from market.price_cache import get_next_price_after
from portfolio.valuation import dt_years, grow_cash, wealth
from portfolio.rebalance import rebalance_to_pi

from market.price_cache import get_prices
from model.returns import compute_log_returns
from model.estimate import estimate_mu_sigma_from_log_returns

from market.alphavantage import fetch_daily_close, fetch_intraday_close, AlphaVantageError
from market.price_cache import insert_prices

from hjb.grid import make_grid
from hjb.solver_fd import solve_hjb_fd, recommend_nt
from hjb.policy import interp_policy
import numpy as np
from market.price_cache import get_latest_price
from cli.commands import cmd_run_daily

from storage.repo import (
    PortfolioState, upsert_state, load_state, log_event, utc_now_iso,
    upsert_equity_point, get_equity_series, get_latest_equity_point,
    insert_rebalance
)

from cli.commands import (
  cmd_init, cmd_show, cmd_load_csv, cmd_show_prices,
  cmd_equity_seed, cmd_equity,
  cmd_update_fixed_pi, cmd_estimate,
  cmd_av_daily, cmd_av_intraday,
  cmd_sync_latest_daily,
  cmd_hjb_test, cmd_update_hjb,
  cmd_backup_db, cmd_reset_state, cmd_reset_all, cmd_init_auto
)
from app.engine import estimate_asof


import pandas as pd
from market.price_cache import insert_prices, get_prices, get_latest_price

from ui.live_plot import run as ui_live_plot


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--symbol", type=str, default="SPY")
    p_init.add_argument("--x0", type=float, default=100.0)
    p_init.add_argument("--price", type=float, default=100.0)
    p_init.add_argument("--r", type=float, default=0.02)
    p_init.add_argument("--gamma", type=float, default=3.0)
    p_init.add_argument("--T", type=float, default=1.0)
    p_init.set_defaults(func=cmd_init)

    p_show = sub.add_parser("show")
    p_show.set_defaults(func=cmd_show)

    p_load = sub.add_parser("load_csv")
    p_load.add_argument("--symbol", type=str, required=True)
    p_load.add_argument("--granularity", type=str, choices=["DAILY", "INTRADAY"], required=True)
    p_load.add_argument("--path", type=str, required=True)
    p_load.set_defaults(func=cmd_load_csv)

    p_prices = sub.add_parser("prices")
    p_prices.add_argument("--symbol", type=str, required=True)
    p_prices.add_argument("--granularity", type=str, choices=["DAILY", "INTRADAY"], required=True)
    p_prices.add_argument("--limit", type=int, default=5)
    p_prices.set_defaults(func=cmd_show_prices)

    p_seed = sub.add_parser("equity_seed")
    p_seed.add_argument("--note", type=str, default="SEED")
    p_seed.set_defaults(func=cmd_equity_seed)

    p_eq = sub.add_parser("equity")
    p_eq.add_argument("--symbol", type=str, required=True)
    p_eq.add_argument("--limit", type=int, default=10)
    p_eq.set_defaults(func=cmd_equity)

    p_up = sub.add_parser("update_fixed_pi")
    p_up.add_argument("--pi", type=float, default=0.5)
    p_up.set_defaults(func=cmd_update_fixed_pi)

    p_est = sub.add_parser("estimate")
    p_est.add_argument("--symbol", type=str, required=True)
    p_est.add_argument("--window", type=int, default=60)
    p_est.add_argument("--annual_days", type=int, default=252)
    p_est.set_defaults(func=cmd_estimate)

    p_avd = sub.add_parser("av_daily")
    p_avd.add_argument("--symbol", type=str, required=True)
    p_avd.add_argument("--outputsize", type=str, choices=["compact", "full"], default="compact")
    p_avd.set_defaults(func=cmd_av_daily)

    p_avi = sub.add_parser("av_intraday")
    p_avi.add_argument("--symbol", type=str, required=True)
    p_avi.add_argument("--interval", type=str, choices=["1min","5min","15min","30min","60min"], default="60min")
    p_avi.add_argument("--outputsize", type=str, choices=["compact", "full"], default="compact")
    p_avi.set_defaults(func=cmd_av_intraday)

    p_ht = sub.add_parser("hjb_test")
    p_ht.add_argument("--r", type=float, default=0.02)
    p_ht.add_argument("--mu", type=float, default=0.08)
    p_ht.add_argument("--sigma", type=float, default=0.20)
    p_ht.add_argument("--gamma", type=float, default=3.0)
    p_ht.add_argument("--pi_min", type=float, default=0.0)
    p_ht.add_argument("--pi_max", type=float, default=1.0)
    p_ht.add_argument("--T", type=float, default=1.0)
    p_ht.add_argument("--x_min", type=float, default=10.0)
    p_ht.add_argument("--x_max", type=float, default=500.0)
    p_ht.add_argument("--nx", type=int, default=400)
    p_ht.add_argument("--bc", type=str, choices=["NEUMANN","DIRICHLET"], default="NEUMANN")
    p_ht.set_defaults(func=cmd_hjb_test)

    p_uh = sub.add_parser("update_hjb")
    p_uh.add_argument("--window", type=int, default=60)
    p_uh.add_argument("--annual_days", type=int, default=252)
    p_uh.add_argument("--nx", type=int, default=400)
    p_uh.add_argument("--x_min_factor", type=float, default=0.1)
    p_uh.add_argument("--x_max_factor", type=float, default=3.0)
    p_uh.add_argument("--xmin_floor", type=float, default=1e-3)
    p_uh.add_argument("--bc", type=str, choices=["NEUMANN","DIRICHLET"], default="NEUMANN")
    p_uh.set_defaults(func=cmd_update_hjb)

    p_sync = sub.add_parser("sync_latest_daily")
    p_sync.set_defaults(func=cmd_sync_latest_daily)
    p_rd = sub.add_parser("run_daily")
    p_rd.add_argument("--outputsize", choices=["compact", "full"], default="compact")
    p_rd.add_argument("--window", type=int, default=60)
    p_rd.add_argument("--annual_days", type=int, default=252)
    p_rd.add_argument("--nx", type=int, default=400)
    p_rd.add_argument("--x_min_factor", type=float, default=0.1)
    p_rd.add_argument("--x_max_factor", type=float, default=3.0)
    p_rd.add_argument("--xmin_floor", type=float, default=1e-3)
    p_rd.add_argument("--bc", choices=["NEUMANN", "DIRICHLET"], default="NEUMANN")
    p_rd.add_argument("--max_steps", type=int, default=None)
    p_rd.add_argument("--dry_run", action="store_true")
    p_rd.set_defaults(func=cmd_run_daily)

    p_bk = sub.add_parser("backup_db")
    p_bk.set_defaults(func=cmd_backup_db)

    p_rs = sub.add_parser("reset_state")
    p_rs.add_argument("--force", action="store_true")
    p_rs.set_defaults(func=cmd_reset_state)

    p_ra = sub.add_parser("reset_all")
    p_ra.add_argument("--force", action="store_true")
    p_ra.set_defaults(func=cmd_reset_all)

    p_ia = sub.add_parser("init_auto")
    p_ia.add_argument("--symbol", type=str, required=True)
    p_ia.add_argument("--x0", type=float, required=True)
    p_ia.add_argument("--r", type=float, default=0.02)
    p_ia.add_argument("--gamma", type=float, default=3.0)
    p_ia.add_argument("--T", type=float, default=1.0)
    p_ia.add_argument("--pi_min", type=float, default=0.0)
    p_ia.add_argument("--pi_max", type=float, default=1.0)
    p_ia.add_argument("--outputsize", choices=["compact", "full"], default="compact")
    p_ia.add_argument("--force", action="store_true")
    p_ia.set_defaults(func=cmd_init_auto)

    p_lp = sub.add_parser("live_plot")
    p_lp.add_argument("--symbol", type=str, default=None)
    p_lp.set_defaults(func=cmd_live_plot)


    
    return p

def cmd_live_plot(args):
    ui_live_plot(symbol=args.symbol)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

