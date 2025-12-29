# src/hjb/policy.py
from __future__ import annotations
import numpy as np

def interp_policy(x_grid: np.ndarray, pi_grid: np.ndarray, x: float) -> float:
    xg = np.asarray(x_grid, dtype=float)
    pg = np.asarray(pi_grid, dtype=float)
    return float(np.interp(float(x), xg, pg))
