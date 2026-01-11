from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import queries
from ui.actions import run_daily_now


def render():
    st.header("Tableau de bord")

    symbols = queries.list_symbols()
    if not symbols:
        st.warning("Aucun symbole en base. Fais init_auto d'abord.")
        return

    # Inputs
    colA, colB, colC, colD = st.columns([2, 2, 2, 2])
    with colA:
        symbol = st.selectbox("Symbol", symbols, index=0)
    with colB:
        window = st.number_input("Fenêtre estimation (jours)", min_value=10, max_value=252, value=60, step=5)
    with colC:
        nx = st.number_input("Nx (HJB)", min_value=50, max_value=2000, value=400, step=50)
    with colD:
        bc = st.selectbox("BC", ["NEUMANN", "DIRICHLET"], index=0)

    # Boutons
    if "run_daily_busy" not in st.session_state:
        st.session_state.run_daily_busy = False

    b1, b2 = st.columns([1, 1])
    status_box = st.empty()

    with b1:
        if st.button("Mise à jour maintenant (run_daily)", use_container_width=True, disabled=st.session_state.run_daily_busy):
            st.session_state.run_daily_busy = True
            status_box.info("run_daily en cours… (HTTP + calcul HJB)")

            try:
                res = run_daily_now(
                    window=int(window),
                    annual_days=252,
                    nx=int(nx),
                    x_min_factor=0.1,
                    x_max_factor=3.0,
                    xmin_floor=1e-3,
                    bc=str(bc),
                    timeout_s=180,
                    outputsize="compact",
                )

                if res.timed_out:
                    status_box.error("Timeout (180s). API lente / bloquée ou HJB trop long.")
                elif not res.ok:
                    status_box.error(f"Erreur run_daily (code {res.returncode})")
                else:
                    status_box.success("OK run_daily")

                if res.stdout:
                    st.code(res.stdout)
                if res.stderr:
                    st.code(res.stderr)

            finally:
                st.session_state.run_daily_busy = False
                st.rerun()

    with b2:
        if st.button("Rafraîchir", use_container_width=True):
            st.rerun()

    # Données
    eq = queries.get_equity_curve(symbol, limit=1000)
    rb = queries.get_rebalances(symbol, limit=200)

    # KPIs (prend la dernière ligne equity_curve et rebalances)
    if len(eq) > 0:
        last = eq.iloc[0]
        last_ts = last["ts_utc"]
        last_w = float(last["wealth"])
        last_p = float(last["price"])
    else:
        last_ts = None
        last_w = float("nan")
        last_p = float("nan")

    if len(rb) > 0:
        r0 = rb.iloc[0]
        last_pi = float(r0["pi"])
        last_sig = float(r0["sigma_annual"])
    else:
        last_pi = float("nan")
        last_sig = float("nan")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Dernière date", str(last_ts) if last_ts else "—")
    k2.metric("Wealth", f"{last_w:.4f}" if last_ts else "—")
    k3.metric("Prix", f"{last_p:.2f}" if last_ts else "—")
    k4.metric("pi*", f"{last_pi:.4f}" if last_ts else "—")
    k5.metric("sigma (ann.)", f"{last_sig:.4f}" if last_ts else "—")

    st.divider()

    # Courbe wealth
    if len(eq) > 0:
        df = eq.copy()
        df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.sort_values("ts")
        fig = px.line(df, x="ts", y="wealth", markers=True, title=f"Equity curve — {symbol}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun point dans equity_curve.")

    with st.expander("Dernières lignes (equity_curve)"):
        st.dataframe(eq, use_container_width=True, height=240)

    with st.expander("Dernières lignes (rebalances)"):
        st.dataframe(rb, use_container_width=True, height=240)
