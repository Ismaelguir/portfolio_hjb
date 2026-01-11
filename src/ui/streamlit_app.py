# src/ui/app.py
from __future__ import annotations

import streamlit as st

from ui.state import ensure_session_defaults
from ui.pages import (
    setup_page,
    dashboard_page,
    hjb_page,
    data_page,
)

st.set_page_config(page_title="portfolio_hjb", layout="wide")

ensure_session_defaults()

st.sidebar.title("portfolio_hjb")
page = st.sidebar.radio(
    "Navigation",
    ["Setup", "Tableau de bord", "HJB", "Données"],
    index=1,
)

if page == "Setup":
    setup_page.render()
elif page == "Tableau de bord":
    dashboard_page.render()
elif page == "HJB":
    hjb_page.render()
elif page == "Données":
    data_page.render()
