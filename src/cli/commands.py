# src/cli/commands.py
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from storage.db import connect, init_db
from storage.repo import (
    PortfolioState, upsert_state, load_state, log_event, utc_now_iso,
    upsert_equity_point, get_equity_series, get_latest_equity_point,
    insert_rebalance
)
from market.price_cache import (
    get_next_price_after, get_prices, get_latest_price, insert_prices
)
from market.alphavantage import fetch_daily_close, fetch_intraday_close, AlphaVantageError
from portfolio.valuation import dt_years, grow_cash, wealth
from portfolio.rebalance import rebalance_to_pi
from model.returns import compute_log_returns
from model.estimate import estimate_mu_sigma_from_log_returns
from hjb.grid import make_grid
from hjb.solver_fd import solve_hjb_fd, recommend_nt
from hjb.policy import interp_policy

from app.engine import update_hjb_step
from app.engine import estimate_asof
from app.engine import backup_db_file, reset_state_tables, reset_all_tables

from app.engine import setup_portfolio
from storage.repo import get_run_config


def cmd_init(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    symbol = args.symbol
    r = args.r
    gamma = args.gamma
    T = args.T
    pi_min, pi_max = 0.0, 1.0

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

    dy = dt_years(state.last_ts_utc, new_ts)
    cash_grown = grow_cash(state.cash_risk_free, state.risk_free_rate, dy)
    x_before = wealth(state.shares_risky, new_price, cash_grown)

    res = rebalance_to_pi(
        wealth=x_before,
        price=new_price,
        pi=args.pi,
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

    # 2) estimation mu/sigma annualisés
    window = args.window
    annual_days = args.annual_days
    est = estimate_asof(
    conn,
    state.symbol,
    new_ts,
    window=window,
    annual_days=annual_days,
    sigma_method=args.sigma_method,
    ewma_lambda=args.ewma_lambda,
    gbm_correction=not args.no_gbm_correction,
)

    if est is None:
        raise SystemExit("Pas assez d'historique DAILY pour estimer mu/sigma (augmente les prix en base).")

    mu_a = est.mean_annual
    sig_a = est.vol_annual

    # 3) solveur HJB
    x_min = max(args.xmin_floor, args.x_min_factor * x_before)
    x_max = args.x_max_factor * x_before
    nx = args.nx
    T = state.horizon_years
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
        sigma_method=args.sigma_method,
        ewma_lambda=(None if args.sigma_method == "std" else args.ewma_lambda),
        gbm_correction=(not args.no_gbm_correction),

    )

    log_event(conn, "UPDATE_HJB", {"ts": new_ts, "price": new_price, "pi": pi_star, "mu": mu_a, "sigma": sig_a})
    print("OK update_hjb:", new_ts, "price", new_price, "pi*", pi_star)


def cmd_estimate(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    prices = get_prices(conn, args.symbol, "DAILY")
    if prices.empty:
        raise SystemExit("Aucun prix DAILY en base pour ce symbol.")

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

    est = estimate_mu_sigma_from_log_returns(
    rets["log_return"],
    trading_days_per_year=args.annual_days,
    sigma_method=args.sigma_method,
    ewma_lambda=args.ewma_lambda,
    gbm_correction=not args.no_gbm_correction,
)


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
            "sigma_method": args.sigma_method,
            "ewma_lambda": (None if args.sigma_method == "std" else args.ewma_lambda),
            "gbm_correction": (not args.no_gbm_correction),
        },
    )

    print(f"Estimation (asof {asof_ts}) sur {args.window} rendements log:")
    print(f"  mu_daily    = {est.mean_daily:.8f}")
    print(f"  sigma_daily = {est.vol_daily:.8f}")
    print(f"  mu_annual_log = {est.mean_annual_log:.6f}")
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
    try:
        rows = fetch_intraday_close(args.symbol, interval=args.interval, outputsize=args.outputsize)
        n = insert_prices(conn, args.symbol, "INTRADAY", rows, source="ALPHAVANTAGE")
        log_event(conn, "AV_INTRADAY", {"symbol": args.symbol, "n": n, "interval": args.interval, "outputsize": args.outputsize})
        print(f"OK AV_INTRADAY: {n} points (INTRADAY) insérés pour {args.symbol} ({args.interval})")
    except AlphaVantageError as e:
        print(f"INTRADAY indisponible via Alpha Vantage (premium): {e}")
        return


def cmd_hjb_test(args: argparse.Namespace) -> None:
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
        shares_risky=state.shares_risky,
        cash_risk_free=cash_grown,
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



# src/cli/commands.py



def cmd_run_daily(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    state = load_state(conn)
    if state is None:
        raise SystemExit("Aucun état. Lance d'abord init.")

    # 1) rafraîchir les prix DAILY
    try:
        rows = fetch_daily_close(state.symbol, outputsize=args.outputsize)
    except AlphaVantageError as e:
        raise SystemExit(f"Échec Alpha Vantage (daily): {e}")

    inserted = insert_prices(conn, state.symbol, "DAILY", rows, source="ALPHAVANTAGE")
    log_event(conn, "RUN_DAILY_REFRESH", {"symbol": state.symbol, "inserted": inserted})
    print(f"OK refresh daily: {inserted} points insérés (éventuels upserts).")

    # 2) rattrapage jour par jour
    steps = 0
    while True:
        if args.max_steps is not None and steps >= args.max_steps:
            print(f"STOP: max_steps={args.max_steps} atteint.")
            break

        nxt = get_next_price_after(conn, state.symbol, "DAILY", state.last_ts_utc)
        if nxt is None:
            if steps == 0:
                print("Rien à faire: aucun nouveau DAILY après last_ts_utc.")
            else:
                print(f"Catch-up terminé: {steps} mise(s) à jour appliquée(s).")
            break

        new_ts, new_price = nxt

        if args.dry_run:
            print(f"[DRY] appliquerait update_hjb au {new_ts} (price={new_price})")
            # on simule l'avancement du pointeur pour éviter boucle infinie en dry-run
            # mais sans modifier la base, on ne peut pas “sauver” state.
            state = state.__class__(**{**state.__dict__, "last_ts_utc": new_ts})
            steps += 1
            continue

        res = update_hjb_step(
            conn=conn,
            state=state,
            new_ts=new_ts,
            new_price=float(new_price),
            window=args.window,
            annual_days=args.annual_days,
            nx=args.nx,
            x_min_factor=args.x_min_factor,
            x_max_factor=args.x_max_factor,
            xmin_floor=args.xmin_floor,
            bc=args.bc,
            sigma_method=args.sigma_method,
            ewma_lambda=args.ewma_lambda,
            gbm_correction=not args.no_gbm_correction,
        )

        state = res.new_state
        steps += 1
        print(f"OK {steps}: {res.ts_utc} price={res.price} pi*={res.pi_star:.6f} wealth={state.last_wealth:.6f}")



def cmd_backup_db(args: argparse.Namespace) -> None:
    path = backup_db_file()
    print(f"OK backup: {path}")


def cmd_reset_state(args: argparse.Namespace) -> None:
    if not args.force:
        print("Refus: ajoute --force pour supprimer état + historique (prices conservés).")
        return
    conn = connect()
    init_db(conn)
    reset_state_tables(conn)
    print("OK reset_state: état + historique supprimés, prices conservés.")


def cmd_reset_all(args: argparse.Namespace) -> None:
    if not args.force:
        print("Refus: ajoute --force pour supprimer TOUT (y compris prices).")
        return
    conn = connect()
    init_db(conn)
    reset_all_tables(conn)
    print("OK reset_all: toutes les tables ont été vidées.")


def cmd_init_auto(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    existing = load_state(conn)
    if existing is not None and not args.force:
        print("Un état existe déjà. Utilise --force (ou fais reset_state --force).")
        return

    # 1) rafraîchir prices daily
    try:
        rows = fetch_daily_close(args.symbol, outputsize=args.outputsize)
    except AlphaVantageError as e:
        raise SystemExit(f"Échec Alpha Vantage (daily): {e}")

    insert_prices(conn, args.symbol, "DAILY", rows, source="ALPHAVANTAGE")

    # 2) prendre le dernier prix dispo
    latest = get_latest_price(conn, args.symbol, "DAILY")
    if latest is None:
        raise SystemExit("Aucun prix DAILY disponible après téléchargement.")
    ts, price = latest

    # 3) initialiser l’état (tout en cash au départ)
    st = PortfolioState(
        symbol=args.symbol,
        risk_free_rate=args.r,
        gamma=args.gamma,
        horizon_years=args.T,
        pi_min=args.pi_min,
        pi_max=args.pi_max,
        shares_risky=0.0,
        cash_risk_free=float(args.x0),
        last_ts_utc=ts,
        last_price=float(price),
        last_wealth=float(args.x0),
    )
    upsert_state(conn, st)

    upsert_equity_point(
        conn=conn,
        symbol=st.symbol,
        ts_utc=st.last_ts_utc,
        wealth=st.last_wealth,
        price=st.last_price,
        shares_risky=st.shares_risky,
        cash_risk_free=st.cash_risk_free,
        note="INIT_AUTO",
    )

    log_event(conn, "INIT_AUTO", {"symbol": st.symbol, "ts": ts, "price": float(price), "x0": float(args.x0)})
    print("OK init_auto:", st)


def cmd_setup(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)

    st = setup_portfolio(
        conn,
        symbol=args.symbol,
        x0=args.x0,
        r=args.r,
        gamma=args.gamma,
        T=args.T,
        pi_min=args.pi_min,
        pi_max=args.pi_max,
        mu_window=args.mu_window,
        sigma_window=args.sigma_window,
        annual_days=args.annual_days,
        sigma_method=args.sigma_method,
        ewma_lambda=args.ewma_lambda,
        gbm_correction=(not args.no_gbm_correction),
        nx=args.nx,
        bc=args.bc,
        x_min_factor=args.x_min_factor,
        x_max_factor=args.x_max_factor,
        xmin_floor=args.xmin_floor,
        fetch_prices=(not args.no_fetch_prices),
        outputsize=args.outputsize,
    )
    print("OK setup:", st)
