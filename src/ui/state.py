import streamlit as st

def ensure_session_defaults():
    st.session_state.setdefault("symbol", None)
    st.session_state.setdefault("refresh_minutes", 60)
