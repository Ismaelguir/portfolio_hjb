# src/hjb/solver_fd.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .utility import crra_utility
from .boundary import apply_neumann_zero_gradient, apply_dirichlet

@dataclass(frozen=True)
class HJBResult:
    grid_x: np.ndarray          # (nx,)
    grid_t: np.ndarray          # (nt+1,)
    V0: np.ndarray              # V(t=0, x_i)
    pi0: np.ndarray             # pi*(t=0, x_i)
    V: np.ndarray               # (nt+1, nx) optionnel pour debug
    pi: np.ndarray              # (nt+1, nx) optionnel pour debug

def _derivatives_centered(V: np.ndarray, dx: float):
    # V: (nx,)
    Vx = np.zeros_like(V)
    Vxx = np.zeros_like(V)

    Vx[1:-1] = (V[2:] - V[:-2]) / (2.0 * dx)
    Vxx[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / (dx * dx)
    return Vx, Vxx

def solve_hjb_fd(
    x: np.ndarray,
    t: np.ndarray,
    r: float,
    mu: float,
    sigma: float,
    gamma: float,
    pi_min: float,
    pi_max: float,
    bc: str = "NEUMANN",  # "NEUMANN" ou "DIRICHLET"
) -> HJBResult:
    """
    Schéma explicite backward :
      V^n = V^{n+1} - dt * H(V_x^{n+1}, V_xx^{n+1})
    où H utilise pi* dérivé de V^{n+1}.
    """
    nx = x.size
    nt = t.size - 1
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    V = np.zeros((nt + 1, nx), dtype=float)
    pi = np.zeros((nt + 1, nx), dtype=float)

    # condition terminale
    V[-1, :] = crra_utility(x, gamma)

    # valeurs Dirichlet si choisi (approx : on fixe V(t, xmin/xmax) = U(xmin/xmax))
    left_dir = float(crra_utility(np.array([x[0]]), gamma)[0])
    right_dir = float(crra_utility(np.array([x[-1]]), gamma)[0])

    for n in range(nt - 1, -1, -1):
        Vn1 = V[n + 1, :].copy()

        # conditions aux bords sur V^{n+1} pour dérivées
        if bc.upper() == "NEUMANN":
            apply_neumann_zero_gradient(Vn1)
        else:
            apply_dirichlet(Vn1, left_dir, right_dir)

        Vx, Vxx = _derivatives_centered(Vn1, dx)

        # pi* sur l'intérieur (on laisse bords = clamp 0)
        denom = (sigma * sigma) * x * Vxx
        numer = -(mu - r) * Vx

        # éviter division par 0 / mauvaise concavité
        pi_raw = np.zeros_like(x)
        mask = (np.abs(denom) > 1e-14)
        pi_raw[mask] = numer[mask] / denom[mask]

        pi_clipped = np.clip(pi_raw, pi_min, pi_max)
        pi[n + 1, :] = pi_clipped  # politique associée au niveau n+1

        # Hamiltonien H = drift + diffusion
        drift = (r + pi_clipped * (mu - r)) * x * Vx
        diff = 0.5 * (pi_clipped * sigma * x) ** 2 * Vxx

        Vn = Vn1 + dt * (drift + diff)

        # appliquer BC sur V^n
        if bc.upper() == "NEUMANN":
            apply_neumann_zero_gradient(Vn)
        else:
            apply_dirichlet(Vn, left_dir, right_dir)

        V[n, :] = Vn

    # pi à t=0 (on peut le recalculer à partir de V[0], mais on prend pi[1] ≈ pi(t=dt))
    # Mieux : recalcul direct sur V[0]
    V0 = V[0, :].copy()
    V0_for_der = V0.copy()
    if bc.upper() == "NEUMANN":
        apply_neumann_zero_gradient(V0_for_der)
    else:
        apply_dirichlet(V0_for_der, left_dir, right_dir)
    Vx0, Vxx0 = _derivatives_centered(V0_for_der, dx)
    denom0 = (sigma * sigma) * x * Vxx0
    numer0 = -(mu - r) * Vx0
    pi0 = np.zeros_like(x)
    mask0 = (np.abs(denom0) > 1e-14)
    pi0[mask0] = numer0[mask0] / denom0[mask0]
    pi0 = np.clip(pi0, pi_min, pi_max)

    return HJBResult(grid_x=x, grid_t=t, V0=V0, pi0=pi0, V=V, pi=pi)

def recommend_nt(T: float, x_min: float, x_max: float, nx: int, sigma: float, pi_max: float) -> int:
    dx = (x_max - x_min) / (nx - 1)
    a_max = (pi_max * sigma * x_max) ** 2  # ordre de grandeur diffusion
    if a_max < 1e-12:
        return max(50, int(np.ceil(T * 50)))
    dt_max = 0.45 * dx * dx / (a_max + 1e-12)  # marge de sécurité
    nt = int(np.ceil(T / dt_max))
    return max(nt, 50)
