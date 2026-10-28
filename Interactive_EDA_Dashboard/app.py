"""Interactive EDA Dashboard - entry point.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from utils.data_loader import SAMPLE_DATASETS, load_sample, load_uploaded
from utils.state import init_state, reset, set_dataset, undo

st.set_page_config(page_title="Interactive EDA Dashboard", page_icon="📊", layout="wide")
init_state()
ss = st.session_state


# ---------- sidebar: data source ----------

def data_source_sidebar() -> None:
    st.sidebar.title("📊 EDA Dashboard")
    source = st.sidebar.radio("Data source", ["Upload file", "Sample dataset"], horizontal=True)

    if source == "Upload file":
        file = st.sidebar.file_uploader("CSV, Excel or JSON file", type=["csv", "tsv", "txt", "xlsx", "xlsm", "json"])
        if file is not None and ss.source_id != f"upload:{file.file_id}":
            try:
                df = load_uploaded(file.getvalue(), file.name)
            except Exception as e:  # show any parsing problem to the user instead of crashing
                st.sidebar.error(f"Could not read **{file.name}**: {e}")
                return
            if df.empty:
                st.sidebar.error("The file has no rows.")
                return
            set_dataset(df, file.name, f"upload:{file.file_id}")
    else:
        name = st.sidebar.selectbox("Choose a sample", list(SAMPLE_DATASETS))
        if ss.source_id != f"sample:{name}":
            set_dataset(load_sample(name), name, f"sample:{name}")

    if ss.df is not None:
        st.sidebar.caption(f"**{ss.dataset_name}**: {ss.df.shape[0]:,} rows × {ss.df.shape[1]} columns")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("↩ Undo", width="stretch", disabled=not ss.history,
                     help="Undo the last cleaning step"):
            undo()
            st.rerun()
        if c2.button("⟲ Reset", width="stretch", disabled=not ss.log,
                     help="Discard all cleaning steps"):
            reset()
            st.rerun()


# ---------- sidebar: row filters ----------

def filters_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Let the user narrow rows down; every page then uses the filtered data."""
    ver = ss.data_version
    with st.sidebar.expander("🔍 Filter rows", expanded=False):
        cols_key = f"filter_cols_{ver}"
        if cols_key in ss:  # drop columns that no longer exist after cleaning
            ss[cols_key] = [c for c in ss[cols_key] if c in df.columns]
        chosen = st.multiselect("Filter by column", list(df.columns), key=cols_key)

        mask = pd.Series(True, index=df.index)
        for col in chosen:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                lo, hi = s.min(), s.max()
                if pd.isna(lo):
                    continue
                picked = st.date_input(col, (lo.date(), hi.date()), min_value=lo.date(),
                                       max_value=hi.date(), key=f"f_{ver}_date_{col}")
                if isinstance(picked, tuple) and len(picked) == 2:
                    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
                    mask &= s.ge(start) & s.lt(end)
            elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                lo, hi = float(s.min()), float(s.max())
                if pd.isna(lo) or lo == hi:
                    st.caption(f"{col}: only one value, nothing to filter")
                    continue
                key = f"f_{ver}_num_{col}"
                if key in ss:  # keep a stored range inside the current min/max
                    a, b = ss[key]
                    ss[key] = (min(max(a, lo), hi), max(min(b, hi), lo))
                a, b = st.slider(col, lo, hi, (lo, hi), key=key)
                if (a, b) != (lo, hi):
                    mask &= s.between(a, b)
            else:
                options = sorted(s.dropna().astype(str).unique())[:500]
                key = f"f_{ver}_cat_{col}"
                if key in ss:
                    ss[key] = [v for v in ss[key] if v in options]
                picked = st.multiselect(f"{col} (keep only)", options, key=key,
                                        placeholder="All values")
                if picked:
                    mask &= s.astype(str).isin(picked)

    view = df[mask]
    if len(view) != len(df):
        st.sidebar.info(f"Filters active: showing **{len(view):,}** of {len(df):,} rows")
    return view


data_source_sidebar()
if ss.df is not None:
    ss.df_view = filters_sidebar(ss.df)

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/overview.py", title="Data Overview", icon="📋"),
    st.Page("pages/data_quality.py", title="Data Quality & Cleaning", icon="🧹"),
    st.Page("pages/univariate.py", title="Univariate Analysis", icon="📈"),
    st.Page("pages/bivariate.py", title="Bivariate & Multivariate", icon="🔗"),
    st.Page("pages/outliers.py", title="Outlier Detection", icon="🎯"),
]
st.navigation(pages).run()
