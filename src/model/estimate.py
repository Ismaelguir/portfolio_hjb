# src/model/estimate.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EstimateResult:
    """Résultat d'estimation.

    Notes:
      - mean_daily est la moyenne des log-rendements journaliers.
      - mean_annual_log = D * mean_daily (drift des log-prix annualisé).
      - mean_annual (si gbm_correction=True) correspond au drift du PRIX sous GBM,
        approx: mu = mean_annual_log + 0.5 * sigma_annual^2.
    """

    window_returns: int
    mean_daily: float
    vol_daily: float
    mean_annual_log: float
    mean_annual: float
    vol_annual: float
    sigma_method: str
    ewma_lambda: float | None


def _sigma_std_daily(log_returns: pd.Series) -> float:
    r = pd.Series(log_returns).astype(float).dropna()
    if r.shape[0] < 2:
        raise ValueError("Pas assez de rendements pour estimer sigma (besoin >= 2)")
    return float(r.std(ddof=1))


def _sigma_ewma_daily(log_returns: pd.Series, lam: float = 0.94) -> float:
    """EWMA (RiskMetrics) sur rendements journaliers.

    Formule:
      sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2

    - Pas de centrage (classique RiskMetrics).
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("ewma_lambda doit être dans (0,1)")

    r = pd.Series(log_returns).astype(float).dropna().to_numpy(dtype=float)
    n = int(r.shape[0])
    if n < 2:
        raise ValueError("Pas assez de rendements pour estimer sigma (besoin >= 2)")

    # init variance: variance échantillon sur toute la série dispo
    v = float(np.var(r, ddof=1))
    for k in range(n):
        v = lam * v + (1.0 - lam) * float(r[k] * r[k])
    return float(np.sqrt(max(v, 0.0)))


def estimate_mu_sigma_from_log_returns(
    log_returns: pd.Series,
    trading_days_per_year: int = 252,
    *,
    sigma_method: str = "std",
    ewma_lambda: float = 0.94,
    gbm_correction: bool = True,
) -> EstimateResult:
    """Estime (mu, sigma) à partir de log-rendements journaliers.

    sigma_method:
      - "std": écart-type échantillon (ddof=1)
      - "ewma": volatilité EWMA (RiskMetrics)

    gbm_correction:
      Si True, renvoie mean_annual corrigé pour être cohérent avec GBM:
        mu = mean_annual_log + 0.5 * sigma_annual^2
    """
    r = pd.Series(log_returns).astype(float).dropna()
    n = int(r.shape[0])
    if n < 2:
        raise ValueError("Pas assez de rendements pour estimer (besoin >= 2)")

    mu_d = float(r.mean())

    method = str(sigma_method).strip().lower()
    if method in ("std", "rolling", "sample"):
        sig_d = _sigma_std_daily(r)
        method = "std"
        lam_used: float | None = None
    elif method in ("ewma", "riskmetrics"):
        sig_d = _sigma_ewma_daily(r, lam=ewma_lambda)
        method = "ewma"
        lam_used = float(ewma_lambda)
    else:
        raise ValueError("sigma_method doit être 'std' ou 'ewma'")

    D = float(trading_days_per_year)
    mu_a_log = D * mu_d
    sig_a = float(np.sqrt(D) * sig_d)

    # Correction GBM: E[log-return] = (mu - 0.5*sigma^2) dt
    if gbm_correction:
        mu_a = float(mu_a_log + 0.5 * sig_a * sig_a)
    else:
        mu_a = float(mu_a_log)

    return EstimateResult(
        window_returns=n,
        mean_daily=float(mu_d),
        vol_daily=float(sig_d),
        mean_annual_log=float(mu_a_log),
        mean_annual=float(mu_a),
        vol_annual=float(sig_a),
        sigma_method=method,
        ewma_lambda=lam_used,
    )
