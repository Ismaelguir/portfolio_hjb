# src/ui/pages/_00_setup.py
from __future__ import annotations

import streamlit as st

from storage.db import connect, init_db

import subprocess
import sys
from pathlib import Path

DEFAULT_SYMBOLS = ["SPY", "QQQ", "VTI", "IEUR", "EEM", "IWM"]


def render() -> None:
    st.title("Setup")

    conn = connect()
    init_db(conn)

    with st.form("setup_form", clear_on_submit=False):
        st.subheader("Portefeuille")
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Ticker", DEFAULT_SYMBOLS, index=0)
            custom = st.text_input("Ticker custom (optionnel)", value="")
            if custom.strip():
                symbol = custom.strip().upper()
        with col2:
            x0 = st.number_input("Richesse initiale (x0)", min_value=0.0, value=10000.0, step=1000.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.number_input("Taux sans risque r (annuel)", min_value=0.0, value=0.02, step=0.005, format="%.4f")
        with col2:
            gamma = st.number_input("Gamma (CRRA)", min_value=0.1, value=3.0, step=0.5)
        with col3:
            T = st.number_input("Horizon T (années)", min_value=0.01, value=1.0, step=0.25)

        st.subheader("Contraintes allocation")
        col1, col2 = st.columns(2)
        with col1:
            pi_min = st.number_input("pi_min", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        with col2:
            pi_max = st.number_input("pi_max", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

        st.subheader("Estimation (recommandé : mu long, sigma court)")
        col1, col2, col3 = st.columns(3)
        with col1:
            mu_window = st.number_input("Fenêtre mu (jours)", min_value=10, value=252, step=10)
        with col2:
            sigma_window = st.number_input("Fenêtre sigma (jours)", min_value=10, value=60, step=5)
        with col3:
            annual_days = st.number_input("Annualisation (jours)", min_value=200, value=252, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            sigma_method = st.selectbox("Méthode sigma", ["ewma", "std"], index=0)
        with col2:
            ewma_lambda = st.number_input("Lambda EWMA", min_value=0.50, max_value=0.999, value=0.94, step=0.01, format="%.3f")
        with col3:
            gbm_correction = st.checkbox("Correction GBM (+0.5 sigma²)", value=True)

        st.subheader("HJB / grille (advanced)")
        col1, col2, col3 = st.columns(3)
        with col1:
            nx = st.number_input("nx", min_value=50, value=400, step=50)
        with col2:
            bc = st.selectbox("BC", ["NEUMANN", "DIRICHLET"], index=0)
        with col3:
            outputsize = st.selectbox("AlphaVantage outputsize", ["compact", "full"], index=0)

        col1, col2, col3 = st.columns(3)
        with col1:
            x_min_factor = st.number_input("x_min_factor", min_value=0.01, value=0.1, step=0.01)
        with col2:
            x_max_factor = st.number_input("x_max_factor", min_value=1.0, value=3.0, step=0.1)
        with col3:
            xmin_floor = st.number_input("xmin_floor", min_value=0.0, value=1e-3, step=1e-3, format="%.6f")

        st.subheader("Prix")
        fetch_prices = st.checkbox("Télécharger / rafraîchir les prix DAILY", value=True)

        submitted = st.form_submit_button("Initialiser")

    if submitted:
        status = st.status("Initialisation...", expanded=True)

        # python de ton venv (robuste)
        project_root = Path(__file__).resolve().parents[2]  # .../src/ui/pages -> .../src
        project_root = project_root.parent                 # .../ (racine projet)
        py = project_root / ".venv" / "bin" / "python"
        if not py.exists():
            py = Path(sys.executable)

        cmd = [
            str(py), "src/main.py", "setup",
            "--symbol", symbol,
            "--x0", str(float(x0)),
            "--r", str(float(r)),
            "--gamma", str(float(gamma)),
            "--T", str(float(T)),
            "--pi_min", str(float(pi_min)),
            "--pi_max", str(float(pi_max)),
            "--mu_window", str(int(mu_window)),
            "--sigma_window", str(int(sigma_window)),
            "--annual_days", str(int(annual_days)),
            "--sigma_method", str(sigma_method),
            "--nx", str(int(nx)),
            "--bc", str(bc),
            "--x_min_factor", str(float(x_min_factor)),
            "--x_max_factor", str(float(x_max_factor)),
            "--xmin_floor", str(float(xmin_floor)),
            "--outputsize", str(outputsize),
        ]

        # ewma_lambda seulement si ewma
        if sigma_method == "ewma":
            cmd += ["--ewma_lambda", str(float(ewma_lambda))]

        # gbm_correction toggle -> no_gbm_correction flag
        if not gbm_correction:
            cmd += ["--no_gbm_correction"]

        # fetch_prices toggle
        if not fetch_prices:
            cmd += ["--no_fetch_prices"]

        status.write("Commande :")
        status.code(" ".join(cmd))

        try:
            p = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=120,  # ajuste si tu veux
            )
        except subprocess.TimeoutExpired:
            status.error("Timeout (120s). AlphaVantage ou autre blocage.")
            st.stop()

        if p.stdout:
            status.write("stdout:")
            status.code(p.stdout)
        if p.stderr:
            status.write("stderr:")
            status.code(p.stderr)

        if p.returncode != 0:
            status.error(f"Setup échoué (code {p.returncode}).")
            st.stop()

        status.success("OK setup terminé.")
        st.rerun()

