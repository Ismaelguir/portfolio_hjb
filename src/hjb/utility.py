# src/hjb/utility.py
from __future__ import annotations
import numpy as np

def crra_utility(x: np.ndarray, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    eps = 1e-12
    x = np.maximum(x, eps)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(x)
    return (x ** (1.0 - gamma)) / (1.0 - gamma)
