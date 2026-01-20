# src/main.py

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from cli.commands import (
    cmd_init,
    cmd_show,
    cmd_load_csv,
    cmd_show_prices,
    cmd_equity_seed,
    cmd_equity,
    cmd_update_fixed_pi,
    cmd_estimate,
    cmd_av_daily,
    cmd_av_intraday,
    cmd_sync_latest_daily,
    cmd_hjb_test,
    cmd_update_hjb,
    cmd_run_daily,
    cmd_backup_db,
    cmd_reset_state,
    cmd_reset_all,
    cmd_init_auto,
    cmd_setup
)

from ui.live_plot import run as ui_live_plot


def _add_estimation_flags(parser: argparse.ArgumentParser) -> None:
    """Ajoute les flags liés à l'estimation (sigma EWMA + correction GBM)."""
    parser.add_argument(
        "--sigma_method",
        choices=["std", "ewma"],
        default="ewma",
        help="Méthode d'estimation de sigma: std (écart-type) ou ewma (RiskMetrics).",
    )
    parser.add_argument(
        "--ewma_lambda",
        type=float,
        default=0.94,
        help="Lambda EWMA (RiskMetrics), typiquement 0.94 en daily.",
    )
    parser.add_argument(
        "--no_gbm_correction",
        action="store_true",
        help="Désactive la correction GBM: mu = mu_log + 0.5*sigma^2.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- init ---
    p_init = sub.add_parser("init")
    p_init.add_argument("--symbol", type=str, default="SPY")
    p_init.add_argument("--x0", type=float, default=100.0)
    p_init.add_argument("--price", type=float, default=100.0)
    p_init.add_argument("--r", type=float, default=0.02)
    p_init.add_argument("--gamma", type=float, default=3.0)
    p_init.add_argument("--T", type=float, default=1.0)
    p_init.set_defaults(func=cmd_init)

    # --- show ---
    p_show = sub.add_parser("show")
    p_show.set_defaults(func=cmd_show)

    # --- load_csv ---
    p_load = sub.add_parser("load_csv")
    p_load.add_argument("--symbol", type=str, required=True)
    p_load.add_argument("--granularity", type=str, choices=["DAILY", "INTRADAY"], required=True)
    p_load.add_argument("--path", type=str, required=True)
    p_load.set_defaults(func=cmd_load_csv)

    # --- prices ---
    p_prices = sub.add_parser("prices")
    p_prices.add_argument("--symbol", type=str, required=True)
    p_prices.add_argument("--granularity", type=str, choices=["DAILY", "INTRADAY"], required=True)
    p_prices.add_argument("--limit", type=int, default=5)
    p_prices.set_defaults(func=cmd_show_prices)

    # --- equity_seed ---
    p_seed = sub.add_parser("equity_seed")
    p_seed.add_argument("--note", type=str, default="SEED")
    p_seed.set_defaults(func=cmd_equity_seed)

    # --- equity ---
    p_eq = sub.add_parser("equity")
    p_eq.add_argument("--symbol", type=str, required=True)
    p_eq.add_argument("--limit", type=int, default=10)
    p_eq.set_defaults(func=cmd_equity)

    # --- update_fixed_pi ---
    p_up = sub.add_parser("update_fixed_pi")
    p_up.add_argument("--pi", type=float, default=0.5)
    p_up.set_defaults(func=cmd_update_fixed_pi)

    # --- estimate ---
    p_est = sub.add_parser("estimate")
    p_est.add_argument("--symbol", type=str, required=True)
    p_est.add_argument("--window", type=int, default=60)
    p_est.add_argument("--annual_days", type=int, default=252)
    _add_estimation_flags(p_est)
    p_est.set_defaults(func=cmd_estimate)

    # --- av_daily ---
    p_avd = sub.add_parser("av_daily")
    p_avd.add_argument("--symbol", type=str, required=True)
    p_avd.add_argument("--outputsize", type=str, choices=["compact", "full"], default="compact")
    p_avd.set_defaults(func=cmd_av_daily)

    # --- av_intraday ---
    p_avi = sub.add_parser("av_intraday")
    p_avi.add_argument("--symbol", type=str, required=True)
    p_avi.add_argument("--interval", type=str, choices=["1min", "5min", "15min", "30min", "60min"], default="60min")
    p_avi.add_argument("--outputsize", type=str, choices=["compact", "full"], default="compact")
    p_avi.set_defaults(func=cmd_av_intraday)

    # --- hjb_test ---
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
    p_ht.add_argument("--bc", type=str, choices=["NEUMANN", "DIRICHLET"], default="NEUMANN")
    p_ht.set_defaults(func=cmd_hjb_test)

    # --- update_hjb ---
    p_uh = sub.add_parser("update_hjb")
    p_uh.add_argument("--window", type=int, default=60)
    p_uh.add_argument("--annual_days", type=int, default=252)
    p_uh.add_argument("--nx", type=int, default=400)
    p_uh.add_argument("--x_min_factor", type=float, default=0.1)
    p_uh.add_argument("--x_max_factor", type=float, default=3.0)
    p_uh.add_argument("--xmin_floor", type=float, default=1e-3)
    p_uh.add_argument("--bc", type=str, choices=["NEUMANN", "DIRICHLET"], default="NEUMANN")
    _add_estimation_flags(p_uh)
    p_uh.set_defaults(func=cmd_update_hjb)

    # --- sync_latest_daily ---
    p_sync = sub.add_parser("sync_latest_daily")
    p_sync.set_defaults(func=cmd_sync_latest_daily)

    # --- run_daily ---
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
    _add_estimation_flags(p_rd)
    p_rd.set_defaults(func=cmd_run_daily)

    # --- backup_db ---
    p_bk = sub.add_parser("backup_db")
    p_bk.set_defaults(func=cmd_backup_db)

    # --- reset_state ---
    p_rs = sub.add_parser("reset_state")
    p_rs.add_argument("--force", action="store_true")
    p_rs.set_defaults(func=cmd_reset_state)

    # --- reset_all ---
    p_ra = sub.add_parser("reset_all")
    p_ra.add_argument("--force", action="store_true")
    p_ra.set_defaults(func=cmd_reset_all)

    # --- init_auto ---
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

    # --- live_plot ---
    p_lp = sub.add_parser("live_plot")
    p_lp.add_argument("--symbol", type=str, default=None)
    p_lp.set_defaults(func=cmd_live_plot)

    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--symbol", type=str, required=True)
    p_setup.add_argument("--x0", type=float, required=True)

    p_setup.add_argument("--r", type=float, default=0.02)
    p_setup.add_argument("--gamma", type=float, default=3.0)
    p_setup.add_argument("--T", type=float, default=1.0)
    p_setup.add_argument("--pi_min", type=float, default=0.0)
    p_setup.add_argument("--pi_max", type=float, default=1.0)

    # recommended defaults
    p_setup.add_argument("--mu_window", type=int, default=252)
    p_setup.add_argument("--sigma_window", type=int, default=60)
    p_setup.add_argument("--annual_days", type=int, default=252)

    p_setup.add_argument("--sigma_method", choices=["std", "ewma"], default="ewma")
    p_setup.add_argument("--ewma_lambda", type=float, default=0.94)
    p_setup.add_argument("--no_gbm_correction", action="store_true")

    p_setup.add_argument("--nx", type=int, default=400)
    p_setup.add_argument("--bc", choices=["NEUMANN", "DIRICHLET"], default="NEUMANN")
    p_setup.add_argument("--x_min_factor", type=float, default=0.1)
    p_setup.add_argument("--x_max_factor", type=float, default=3.0)
    p_setup.add_argument("--xmin_floor", type=float, default=1e-3)

    p_setup.add_argument("--outputsize", choices=["compact", "full"], default="compact")
    p_setup.add_argument("--no_fetch_prices", action="store_true")

    p_setup.set_defaults(func=cmd_setup)


    return p


def cmd_live_plot(args):
    ui_live_plot(symbol=args.symbol)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
