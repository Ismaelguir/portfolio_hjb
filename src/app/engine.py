# src/app/engine.py

from __future__ import annotations

from dataclasses import dataclass

from market.price_cache import get_next_price_after
from model.returns import compute_log_returns
from model.estimate import estimate_mu_sigma_from_log_returns
from market.price_cache import get_prices
from portfolio.valuation import dt_years, grow_cash, wealth
from portfolio.rebalance import rebalance_to_pi
from storage.repo import upsert_state, upsert_equity_point, insert_rebalance, log_event
from hjb.grid import make_grid
from hjb.solver_fd import solve_hjb_fd, recommend_nt
from hjb.policy import interp_policy


from pathlib import Path
from datetime import datetime
import shutil
import sqlite3

from storage.db import get_db_path

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


@dataclass(frozen=True)
class HJBStepResult:
    new_state: object
    pi_star: float
    mu_annual: float
    sigma_annual: float
    wealth_before: float
    ts_utc: str
    price: float


def update_hjb_step(
    conn,
    state,
    new_ts: str,
    new_price: float,
    window: int,
    annual_days: int,
    nx: int,
    x_min_factor: float,
    x_max_factor: float,
    xmin_floor: float,
    bc: str,
) -> HJBStepResult:
    # 1) valorisation (cash + richesse)
    dy = dt_years(state.last_ts_utc, new_ts)
    cash_grown = grow_cash(state.cash_risk_free, state.risk_free_rate, dy)
    x_before = wealth(state.shares_risky, new_price, cash_grown)

    # 2) estimation mu/sigma annualisés
    est = estimate_asof(conn, state.symbol, new_ts, window=window, annual_days=annual_days)
    if est is None:
        raise RuntimeError("Pas assez d'historique DAILY pour estimer mu/sigma.")
    mu_a = est.mean_annual
    sig_a = est.vol_annual

    # 3) solveur HJB sur une plage autour de X
    x_min = max(xmin_floor, x_min_factor * x_before)
    x_max = x_max_factor * x_before
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
        bc=bc,
    )
    pi_star = interp_policy(hjb.grid_x, hjb.pi0, x_before)

    # 4) rééquilibrage
    res = rebalance_to_pi(
        wealth=x_before,
        price=new_price,
        pi=pi_star,
        pi_min=state.pi_min,
        pi_max=state.pi_max,
    )

    # 5) persistance
    new_state = state.__class__(  # garde ton type PortfolioState sans le réimporter ici
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

    log_event(conn, "UPDATE_HJB_STEP", {
        "ts": new_ts, "price": new_price,
        "pi": pi_star, "mu_annual": mu_a, "sigma_annual": sig_a,
        "nx": nx, "nt": nt, "x_min": x_min, "x_max": x_max, "bc": bc,
        "wealth_before": x_before, "wealth_after": res.wealth,
    })

    return HJBStepResult(
        new_state=new_state,
        pi_star=float(pi_star),
        mu_annual=float(mu_a),
        sigma_annual=float(sig_a),
        wealth_before=float(x_before),
        ts_utc=new_ts,
        price=float(new_price),
    )

# src/app/engine.py



def backup_db_file() -> Path:
    """Copie data/portfolio.db vers data/backups/portfolio_YYYYMMDD_HHMMSS.db"""
    src = get_db_path()
    if not src.exists():
        raise FileNotFoundError(f"Base introuvable: {src}")

    backups_dir = src.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = backups_dir / f"portfolio_{stamp}.db"
    shutil.copy2(src, dst)
    return dst


def reset_state_tables(conn: sqlite3.Connection) -> None:
    """Supprime état + historique, mais conserve prices."""
    conn.execute("DELETE FROM equity_curve;")
    conn.execute("DELETE FROM rebalances;")
    conn.execute("DELETE FROM events;")
    conn.execute("DELETE FROM portfolio_state;")
    conn.commit()


def reset_all_tables(conn: sqlite3.Connection) -> None:
    """Supprime tout, y compris prices."""
    conn.execute("DELETE FROM equity_curve;")
    conn.execute("DELETE FROM rebalances;")
    conn.execute("DELETE FROM events;")
    conn.execute("DELETE FROM portfolio_state;")
    conn.execute("DELETE FROM prices;")
    conn.commit()
