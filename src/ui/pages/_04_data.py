# src/ui/pages/_04_data.py
from __future__ import annotations

import streamlit as st

from ui import queries


def _download_csv(df, filename: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render():
    st.header("Données (SQLite)")

    symbols = queries.list_symbols()
    default_symbol = symbols[0] if symbols else None

    table = st.selectbox(
        "Table",
        ["portfolio_state", "equity_curve", "rebalances", "prices", "events"],
        index=1,
    )

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        symbol = None
        if table in ("prices", "equity_curve", "rebalances"):
            symbol = st.selectbox("Symbol", symbols, index=(symbols.index(default_symbol) if default_symbol in symbols else 0))

    with col2:
        gran = None
        if table == "prices":
            grans = queries.list_granularities(symbol) if symbol else ["DAILY"]
            gran = st.selectbox("Granularité", grans, index=0)

    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=5000, value=300, step=50)

    with col4:
        with st.expander("Filtre dates (optionnel)"):
            start = st.date_input("Start", value=None)
            end = st.date_input("End", value=None)

    # Chargement
    if table == "portfolio_state":
        df = queries.get_state()

    elif table == "equity_curve":
        if not symbol:
            st.warning("Aucun symbol disponible.")
            return
        df = queries.get_equity_curve(symbol, start=start or None, end=end or None, limit=int(limit))

    elif table == "rebalances":
        if not symbol:
            st.warning("Aucun symbol disponible.")
            return
        df = queries.get_rebalances(symbol, limit=int(limit))

    elif table == "prices":
        if not symbol:
            st.warning("Aucun symbol disponible.")
            return
        df = queries.get_prices(symbol, granularity=gran or "DAILY", start=start or None, end=end or None, limit=int(limit))

    else:  # events
        df = queries.get_events(limit=int(limit))

    st.caption(f"{len(df)} ligne(s)")
    st.dataframe(df, use_container_width=True, height=520)

    # Exports
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        _download_csv(df, f"{table}.csv", "Télécharger CSV (vue actuelle)")
    with c2:
        st.write("")  # placeholder
