# src/hjb/boundary.py
from __future__ import annotations
import numpy as np

def apply_neumann_zero_gradient(V: np.ndarray) -> None:
    # dV/dx = 0 -> V[0]=V[1], V[-1]=V[-2]
    V[0] = V[1]
    V[-1] = V[-2]

def apply_dirichlet(V: np.ndarray, left_value: float, right_value: float) -> None:
    V[0] = left_value
    V[-1] = right_value
