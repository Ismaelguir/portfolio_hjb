# src/model/returns.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices: DataFrame avec colonnes ['ts_utc', 'price'], trié croissant.
    Retourne DataFrame avec colonnes ['ts_utc', 'log_return'] aligné sur le timestamp du prix courant.
    """
    if prices.empty:
        return pd.DataFrame(columns=["ts_utc", "log_return"])

    df = prices.copy()
    if "ts_utc" not in df.columns or "price" not in df.columns:
        raise ValueError("prices doit contenir les colonnes ts_utc et price")

    df = df.sort_values("ts_utc").reset_index(drop=True)

    p = df["price"].astype(float)
    if (p <= 0).any():
        raise ValueError("Prix <= 0 détecté (log impossible)")

    lr = np.log(p).diff()

    out = pd.DataFrame({
        "ts_utc": df["ts_utc"].astype(str),
        "log_return": lr.astype(float),
    })

    out = out.dropna().reset_index(drop=True)
    return out
