# src/portfolio/rebalance.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RebalanceResult:
    shares_risky: float
    cash_risk_free: float
    wealth: float  # identique avant/après, si pas de coûts


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rebalance_to_pi(
    wealth: float,
    price: float,
    pi: float,
    pi_min: float,
    pi_max: float,
) -> RebalanceResult:
    """
    Applique la cible pi sur la richesse au prix 'price'.
    n = pi X / S, B = (1-pi) X
    """
    if price <= 0:
        raise ValueError("price <= 0 (impossible de rebalancer)")

    pi_c = clamp(float(pi), float(pi_min), float(pi_max))
    shares = (pi_c * float(wealth)) / float(price)
    cash = (1.0 - pi_c) * float(wealth)
    return RebalanceResult(shares_risky=shares, cash_risk_free=cash, wealth=float(wealth))
