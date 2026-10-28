"""Session-state helpers: the working dataset, undo history and the cleaning log."""
import pandas as pd
import streamlit as st

MAX_HISTORY = 15


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("df_original", None)   # dataset as loaded
    ss.setdefault("df", None)            # working (cleaned) dataset
    ss.setdefault("df_view", None)       # working dataset after sidebar filters
    ss.setdefault("dataset_name", None)
    ss.setdefault("source_id", None)
    ss.setdefault("history", [])         # previous versions of df, for undo
    ss.setdefault("log", [])             # human-readable list of cleaning steps
    ss.setdefault("data_version", 0)     # bumped on new dataset -> resets filter widgets
    ss.setdefault("flash", None)


def set_dataset(df: pd.DataFrame, name: str, source_id: str) -> None:
    ss = st.session_state
    ss.df_original = df
    ss.df = df.copy()
    ss.df_view = ss.df
    ss.dataset_name = name
    ss.source_id = source_id
    ss.history = []
    ss.log = []
    ss.data_version += 1


def apply_change(new_df: pd.DataFrame, message: str) -> None:
    """Replace the working dataset, remembering the old one so it can be undone."""
    ss = st.session_state
    ss.history.append(ss.df)
    ss.history = ss.history[-MAX_HISTORY:]
    ss.df = new_df
    ss.log.append(message)
    ss.flash = message


def undo() -> None:
    ss = st.session_state
    if ss.history:
        ss.df = ss.history.pop()
        ss.flash = f"Undid: {ss.log.pop()}" if ss.log else "Undid last change"


def reset() -> None:
    ss = st.session_state
    ss.df = ss.df_original.copy()
    ss.history = []
    ss.log = []
    ss.flash = "Reset to the original dataset"


def require_data() -> pd.DataFrame:
    """Return the filtered dataset, or stop the page with a hint if nothing is loaded."""
    df = st.session_state.get("df_view")
    if df is None:
        st.info("👈 Upload a dataset or choose a sample dataset in the sidebar to get started.")
        st.stop()
    if df.empty:
        st.warning("The current filters remove every row. Loosen the filters in the sidebar.")
        st.stop()
    return df


def show_flash() -> None:
    msg = st.session_state.get("flash")
    if msg:
        st.toast(msg, icon="✅")
        st.session_state.flash = None
