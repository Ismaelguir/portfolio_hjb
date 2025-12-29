# src/hjb/grid.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Grid1D:
    x: np.ndarray
    t: np.ndarray
    dx: float
    dt: float

def make_grid(x_min: float, x_max: float, nx: int, T: float, nt: int) -> Grid1D:
    x = np.linspace(x_min, x_max, nx, dtype=float)
    t = np.linspace(0.0, T, nt + 1, dtype=float)
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    return Grid1D(x=x, t=t, dx=dx, dt=dt)
