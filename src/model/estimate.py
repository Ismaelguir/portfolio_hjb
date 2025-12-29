# src/model/estimate.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class EstimateResult:
    window_returns: int
    mean_daily: float
    vol_daily: float
    mean_annual: float
    vol_annual: float


def estimate_mu_sigma_from_log_returns(
    log_returns: pd.Series,
    trading_days_per_year: int = 252,
) -> EstimateResult:
    """
    log_returns: série de log-rendements journaliers.
    On estime:
      mu_daily = mean(r)
      sigma_daily = std(r) (écart-type échantillon, ddof=1)
    puis annualisation.
    """
    r = pd.Series(log_returns).astype(float).dropna()
    n = int(r.shape[0])
    if n < 2:
        raise ValueError("Pas assez de rendements pour estimer (besoin >= 2)")

    mu_d = float(r.mean())
    sig_d = float(r.std(ddof=1))

    mu_a = float(trading_days_per_year) * mu_d
    sig_a = np.sqrt(float(trading_days_per_year)) * sig_d

    return EstimateResult(
        window_returns=n,
        mean_daily=mu_d,
        vol_daily=sig_d,
        mean_annual=mu_a,
        vol_annual=float(sig_a),
    )
