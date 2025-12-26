# src/main.py
from __future__ import annotations

import argparse

from storage.db import connect, init_db
from storage.repo import PortfolioState, upsert_state, load_state, log_event, utc_now_iso


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


    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

