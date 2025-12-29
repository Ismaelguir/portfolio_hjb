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


from storage.repo import (
    PortfolioState, upsert_state, load_state, log_event, utc_now_iso,
    upsert_equity_point, get_equity_series, get_latest_equity_point,
    insert_rebalance
)



def cmd_init(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    # Valeurs de test (tu changeras plus tard)
    symbol = args.symbol
    r = args.r
    gamma = args.gamma
    T = args.T
    pi_min, pi_max = 0.0, 1.0

    # Pour l'étape A, on met un prix "dummy" et un portefeuille 100% cash
    last_price = args.price
    shares_risky = 0.0
    cash_risk_free = args.x0
    last_wealth = shares_risky * last_price + cash_risk_free

    state = PortfolioState(
        symbol=symbol,
        risk_free_rate=r,
        gamma=gamma,
        horizon_years=T,
        pi_min=pi_min,
        pi_max=pi_max,
        shares_risky=shares_risky,
        cash_risk_free=cash_risk_free,
        last_ts_utc=utc_now_iso(),
        last_price=last_price,
        last_wealth=last_wealth,
    )

    upsert_state(conn, state)
    log_event(conn, "INIT", {"x0": args.x0, "price": last_price, "symbol": symbol})

    print("Base initialisée et état sauvegardé.")
    print(state)


def cmd_show(_: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)
    state = load_state(conn)
    print("État en base:", state)

import pandas as pd
from market.price_cache import insert_prices, get_prices, get_latest_price

def cmd_load_csv(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    df = pd.read_csv(args.path)
    rows = list(zip(df["ts_utc"].astype(str), df["price"].astype(float)))
    n = insert_prices(conn, args.symbol, args.granularity, rows, source="CSV")

    log_event(conn, "LOAD_CSV_PRICES", {"symbol": args.symbol, "granularity": args.granularity, "n": n})
    print(f"OK: {n} points insérés pour {args.symbol} ({args.granularity}) depuis {args.path}")

def cmd_show_prices(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    latest = get_latest_price(conn, args.symbol, args.granularity)
    df = get_prices(conn, args.symbol, args.granularity)
    print("Latest:", latest)
    print(df.tail(args.limit).to_string(index=False))


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


    
    return p

def cmd_equity_seed(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    state = load_state(conn)
    if state is None:
        raise SystemExit("Aucun état en base. Lance d'abord: python src/main.py init ...")

    upsert_equity_point(
        conn=conn,
        symbol=state.symbol,
        ts_utc=state.last_ts_utc,
        wealth=state.last_wealth,
        price=state.last_price,
        shares_risky=state.shares_risky,
        cash_risk_free=state.cash_risk_free,
        note=args.note,
    )

    log_event(conn, "EQUITY_SEED", {"symbol": state.symbol, "ts": state.last_ts_utc, "wealth": state.last_wealth})
    print("OK: point equity_curve ajouté depuis portfolio_state.")

def cmd_equity(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    latest = get_latest_equity_point(conn, args.symbol)
    df = get_equity_series(conn, args.symbol)
    print("Latest equity:", latest)
    if df.empty:
        print("(vide)")
        return
    print(df.tail(args.limit).to_string(index=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

def cmd_update_fixed_pi(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    state = load_state(conn)
    if state is None:
        raise SystemExit("Aucun état. Lance d'abord init.")

    nxt = get_next_price_after(conn, state.symbol, "DAILY", state.last_ts_utc)
    if nxt is None:
        raise SystemExit(
            "Aucun nouveau prix DAILY après last_ts_utc. "
            "Ajoute des prix (load_csv) avec des dates plus récentes."
        )

    new_ts, new_price = nxt

    # 1) faire croître le cash sans risque depuis last_ts -> new_ts
    dy = dt_years(state.last_ts_utc, new_ts)
    cash_grown = grow_cash(state.cash_risk_free, state.risk_free_rate, dy)

    # 2) richesse avant rééquilibrage au nouveau prix
    x_before = wealth(state.shares_risky, new_price, cash_grown)

    # 3) rééquilibrage à pi fixe
    res = rebalance_to_pi(
        wealth=x_before,
        price=new_price,
        pi=args.pi,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
    )

    # 4) persistance nouvel état
    new_state = PortfolioState(
        symbol=state.symbol,
        risk_free_rate=state.risk_free_rate,
        gamma=state.gamma,
        horizon_years=state.horizon_years,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
        shares_risky=res.shares_risky,
        cash_risk_free=res.cash_risk_free,
        last_ts_utc=new_ts,
        last_price=new_price,
        last_wealth=res.wealth,
    )
    upsert_state(conn, new_state)

    # 5) écrire un point de courbe de richesse (après rééquilibrage)
    upsert_equity_point(
        conn=conn,
        symbol=new_state.symbol,
        ts_utc=new_state.last_ts_utc,
        wealth=new_state.last_wealth,
        price=new_state.last_price,
        shares_risky=new_state.shares_risky,
        cash_risk_free=new_state.cash_risk_free,
        note=f"REBAL_PI_{args.pi}",
    )

    log_event(
        conn,
        "UPDATE_FIXED_PI",
        {
            "from_ts": state.last_ts_utc,
            "to_ts": new_ts,
            "price": new_price,
            "dt_years": dy,
            "x_before": x_before,
            "x_after": res.wealth,
            "pi": args.pi,
            "shares": res.shares_risky,
            "cash": res.cash_risk_free,
        },
    )

    print("OK update_fixed_pi")
    print("new_ts:", new_ts, "price:", new_price)
    print("wealth:", res.wealth, "shares:", res.shares_risky, "cash:", res.cash_risk_free)


def estimate_asof(conn, symbol: str, asof_ts: str, window: int, annual_days: int):
    prices = get_prices(conn, symbol, "DAILY")
    prices = prices[prices["ts_utc"] <= asof_ts]
    need = window + 1
    if len(prices) < need:
        return None
    window_prices = prices.tail(need).reset_index(drop=True)
    rets = compute_log_returns(window_prices)
    if len(rets) != window:
        return None
    return estimate_mu_sigma_from_log_returns(rets["log_return"], trading_days_per_year=annual_days)


def cmd_update_hjb(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    state = load_state(conn)
    if state is None:
        raise SystemExit("Aucun état. Lance d'abord init.")

    nxt = get_next_price_after(conn, state.symbol, "DAILY", state.last_ts_utc)
    if nxt is None:
        raise SystemExit("Aucun nouveau prix DAILY après last_ts_utc.")

    new_ts, new_price = nxt

    # 1) valorisation
    dy = dt_years(state.last_ts_utc, new_ts)
    cash_grown = grow_cash(state.cash_risk_free, state.risk_free_rate, dy)
    x_before = wealth(state.shares_risky, new_price, cash_grown)

    # 2) estimation mu/sigma annualisés (peut être None)
    window = args.window
    annual_days = args.annual_days
    est = estimate_asof(conn, state.symbol, new_ts, window=window, annual_days=annual_days)
    if est is None:
        raise SystemExit("Pas assez d'historique DAILY pour estimer mu/sigma (augmente les prix en base).")

    mu_a = est.mean_annual
    sig_a = est.vol_annual

    # 3) solveur HJB sur une fenêtre de richesse autour de X
    x_min = max(args.xmin_floor, args.x_min_factor * x_before)
    x_max = args.x_max_factor * x_before
    nx = args.nx
    T = state.horizon_years  # horizon fixe de ton modèle
    nt = recommend_nt(T, x_min, x_max, nx, sig_a, state.pi_max)
    grid = make_grid(x_min, x_max, nx, T, nt)

    hjb = solve_hjb_fd(
        x=grid.x, t=grid.t,
        r=state.risk_free_rate,
        mu=mu_a,
        sigma=sig_a,
        gamma=state.gamma,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
        bc=args.bc
    )

    pi_star = interp_policy(hjb.grid_x, hjb.pi0, x_before)

    # 4) rebalance
    res = rebalance_to_pi(
        wealth=x_before,
        price=new_price,
        pi=pi_star,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
    )

    new_state = PortfolioState(
        symbol=state.symbol,
        risk_free_rate=state.risk_free_rate,
        gamma=state.gamma,
        horizon_years=state.horizon_years,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
        shares_risky=res.shares_risky,
        cash_risk_free=res.cash_risk_free,
        last_ts_utc=new_ts,
        last_price=new_price,
        last_wealth=res.wealth,
    )
    upsert_state(conn, new_state)

    upsert_equity_point(
        conn=conn,
        symbol=new_state.symbol,
        ts_utc=new_state.last_ts_utc,
        wealth=new_state.last_wealth,
        price=new_state.last_price,
        shares_risky=new_state.shares_risky,
        cash_risk_free=new_state.cash_risk_free,
        note="REBAL_HJB",
    )

    insert_rebalance(
        conn=conn,
        symbol=new_state.symbol,
        ts_utc=new_ts,
        price=new_price,
        wealth_before=x_before,
        wealth_after=res.wealth,
        shares_risky=res.shares_risky,
        cash_risk_free=res.cash_risk_free,
        pi=pi_star,
        mu_annual=mu_a,
        sigma_annual=sig_a,
        window=window,
        annual_days=annual_days,
        note="HJB",
    )

    log_event(conn, "UPDATE_HJB", {"ts": new_ts, "price": new_price, "pi": pi_star, "mu": mu_a, "sigma": sig_a})
    print("OK update_hjb:", new_ts, "price", new_price, "pi*", pi_star)


def cmd_estimate(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    # On récupère les prix DAILY depuis la DB
    prices = get_prices(conn, args.symbol, "DAILY")
    if prices.empty:
        raise SystemExit("Aucun prix DAILY en base pour ce symbol.")

    # Fenêtre: besoin de window+1 prix pour avoir window rendements
    need_prices = args.window + 1
    if len(prices) < need_prices:
        raise SystemExit(
            f"Pas assez de prix: {len(prices)} disponibles, besoin de {need_prices} "
            f"pour window={args.window}."
        )

    window_prices = prices.tail(need_prices).reset_index(drop=True)

    rets = compute_log_returns(window_prices)
    if len(rets) != args.window:
        raise SystemExit(
            f"Incohérence: {len(rets)} rendements obtenus, attendu {args.window}."
        )

    est = estimate_mu_sigma_from_log_returns(rets["log_return"], trading_days_per_year=args.annual_days)

    asof_ts = window_prices["ts_utc"].iloc[-1]
    log_event(
        conn,
        "ESTIMATE_PARAMS",
        {
            "symbol": args.symbol,
            "asof_ts_utc": str(asof_ts),
            "window": args.window,
            "annual_days": args.annual_days,
            "mu_daily": est.mean_daily,
            "sigma_daily": est.vol_daily,
            "mu_annual": est.mean_annual,
            "sigma_annual": est.vol_annual,
        },
    )

    print(f"Estimation (asof {asof_ts}) sur {args.window} rendements log:")
    print(f"  mu_daily    = {est.mean_daily:.8f}")
    print(f"  sigma_daily = {est.vol_daily:.8f}")
    print(f"  mu_annual   = {est.mean_annual:.6f}")
    print(f"  sigma_annual= {est.vol_annual:.6f}")

def cmd_av_daily(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)
    rows = fetch_daily_close(args.symbol, outputsize=args.outputsize)
    n = insert_prices(conn, args.symbol, "DAILY", rows, source="ALPHAVANTAGE")
    log_event(conn, "AV_DAILY", {"symbol": args.symbol, "n": n, "outputsize": args.outputsize})
    print(f"OK AV_DAILY: {n} points (DAILY) insérés pour {args.symbol}")

def cmd_av_intraday(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)
    rows = fetch_intraday_close(args.symbol, interval=args.interval, outputsize=args.outputsize)
    n = insert_prices(conn, args.symbol, "INTRADAY", rows, source="ALPHAVANTAGE")
    log_event(conn, "AV_INTRADAY", {"symbol": args.symbol, "n": n, "interval": args.interval, "outputsize": args.outputsize})
    print(f"OK AV_INTRADAY: {n} points (INTRADAY) insérés pour {args.symbol} ({args.interval})")
    try:
        rows = fetch_intraday_close(...)
    except AlphaVantageError as e:
        print(f"INTRADAY indisponible via Alpha Vantage (premium): {e}")
        return

def cmd_hjb_test(args: argparse.Namespace) -> None:
    # paramètres "simples" test
    r, mu, sigma, gamma = args.r, args.mu, args.sigma, args.gamma
    pi_min, pi_max = args.pi_min, args.pi_max
    T = args.T

    x_min, x_max = args.x_min, args.x_max
    nx = args.nx
    nt = recommend_nt(T, x_min, x_max, nx, sigma, pi_max)
    grid = make_grid(x_min, x_max, nx, T, nt)

    res = solve_hjb_fd(
        x=grid.x, t=grid.t,
        r=r, mu=mu, sigma=sigma, gamma=gamma,
        pi_min=pi_min, pi_max=pi_max,
        bc=args.bc
    )

    pi_merton = (mu - r) / (gamma * sigma * sigma)
    pi_merton = float(np.clip(pi_merton, pi_min, pi_max))

    # mesure simple : moyenne/écart-type de pi0 sur la zone centrale (éviter bords)
    mid = res.pi0[int(0.1*nx):int(0.9*nx)]
    print("pi_merton (clippé) =", pi_merton)
    print("pi0 moyenne (zone centrale) =", float(mid.mean()))
    print("pi0 std (zone centrale)    =", float(mid.std()))
    print("nx =", nx, "nt =", nt, "dx =", grid.dx, "dt =", grid.dt)
    a_max = (pi_max * sigma * x_max) ** 2
    print("stability ratio ~ dt * a_max / dx^2 =", grid.dt * a_max / (grid.dx * grid.dx))

def cmd_sync_latest_daily(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    state = load_state(conn)
    if state is None:
        raise SystemExit("Aucun état. Lance d'abord init.")

    latest = get_latest_price(conn, state.symbol, "DAILY")
    if latest is None:
        raise SystemExit("Aucun prix DAILY en base pour ce symbol.")

    ts, price = latest

    # On valorise à ce ts (croissance du cash) et on recalcule la richesse
    dy = dt_years(state.last_ts_utc, ts)
    cash_grown = grow_cash(state.cash_risk_free, state.risk_free_rate, dy)
    x_now = wealth(state.shares_risky, price, cash_grown)

    new_state = PortfolioState(
        symbol=state.symbol,
        risk_free_rate=state.risk_free_rate,
        gamma=state.gamma,
        horizon_years=state.horizon_years,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
        shares_risky=state.shares_risky,     # inchangé
        cash_risk_free=cash_grown,           # cash mis à jour
        last_ts_utc=ts,
        last_price=price,
        last_wealth=x_now,
    )
    upsert_state(conn, new_state)

    upsert_equity_point(
        conn=conn,
        symbol=new_state.symbol,
        ts_utc=new_state.last_ts_utc,
        wealth=new_state.last_wealth,
        price=new_state.last_price,
        shares_risky=new_state.shares_risky,
        cash_risk_free=new_state.cash_risk_free,
        note="SYNC_LATEST_DAILY",
    )

    log_event(conn, "SYNC_LATEST_DAILY", {"symbol": state.symbol, "ts": ts, "price": price})
    print("OK sync:", ts, "price", price, "wealth", x_now)


if __name__ == "__main__":
    main()

