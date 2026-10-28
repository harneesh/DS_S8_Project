import streamlit as st

from utils.state import show_flash

show_flash()
st.title("📊 Interactive EDA Dashboard")
st.markdown(
    "Explore **any dataset** without writing code: upload a CSV, Excel or JSON file "
    "(or pick a sample) in the sidebar, then use the pages on the left."
)

c1, c2, c3 = st.columns(3)
c1.markdown("#### 📋 Data Overview\nShape, column types, summary statistics and automatic insights.")
c2.markdown("#### 🧹 Data Quality & Cleaning\nMissing values, duplicates, type fixes, undo, and CSV export.")
c3.markdown("#### 📈 Univariate Analysis\nDistribution of one column: histogram, box plot, Q-Q plot, value counts.")
c4, c5, c6 = st.columns(3)
c4.markdown("#### 🔗 Bivariate & Multivariate\nScatter plots, correlation heatmap, group comparisons, statistical tests.")
c5.markdown("#### 🎯 Outlier Detection\nIQR and Z-score methods, visual highlighting, remove or cap outliers.")
c6.markdown("#### 🔍 Filters\nNarrow down rows in the sidebar; every chart updates instantly.")

df = st.session_state.get("df_view")
st.divider()
if df is None:
    st.info("👈 Start by uploading a file or choosing a sample dataset in the sidebar.")
else:
    st.subheader(f"Current dataset: {st.session_state.dataset_name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing cells", f"{df.isna().mean().mean():.1%}")
    m4.metric("Duplicate rows", f"{df.duplicated().sum():,}")
    st.dataframe(df.head(10), width="stretch")
