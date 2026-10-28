import streamlit as st

from utils.analysis import column_info, generate_insights
from utils.data_loader import categorical_cols, datetime_cols, numeric_cols
from utils.state import require_data, show_flash

show_flash()
df = require_data()
st.title("📋 Data Overview")

m = st.columns(6)
m[0].metric("Rows", f"{len(df):,}")
m[1].metric("Columns", df.shape[1])
m[2].metric("Numeric", len(numeric_cols(df)))
m[3].metric("Categorical", len(categorical_cols(df)))
m[4].metric("Date/time", len(datetime_cols(df)))
mem = df.memory_usage(deep=True).sum() / 1024
m[5].metric("Memory", f"{mem / 1024:.1f} MB" if mem >= 1024 else f"{mem:.0f} KB")

st.subheader("💡 Automatic insights")
insights = generate_insights(df)
if not insights:
    st.success("No obvious data issues found.")
for level, msg in insights:
    (st.warning if level == "warning" else st.info)(msg)

st.subheader("Data preview")
c1, c2 = st.columns([1, 3])
mode = c1.radio("Show", ["First rows", "Last rows", "Random sample"])
n = c2.slider("Number of rows", 5, 100, 10)
preview = {"First rows": df.head(n), "Last rows": df.tail(n),
           "Random sample": df.sample(min(n, len(df)), random_state=0)}[mode]
st.dataframe(preview, width="stretch")

st.subheader("Column information")
st.dataframe(column_info(df), width="stretch", hide_index=True,
             column_config={"Missing %": st.column_config.ProgressColumn(
                 "Missing %", min_value=0, max_value=100, format="%.1f%%")})

st.subheader("Summary statistics")
tab_num, tab_cat = st.tabs(["Numeric columns", "Categorical columns"])
with tab_num:
    if numeric_cols(df):
        st.dataframe(df[numeric_cols(df)].describe().T.round(3), width="stretch")
    else:
        st.caption("No numeric columns.")
with tab_cat:
    if categorical_cols(df):
        desc = df[categorical_cols(df)].astype("string").describe().T
        desc["freq %"] = (desc["freq"].astype(float) / desc["count"].astype(float) * 100).round(1)
        st.dataframe(desc, width="stretch")
    else:
        st.caption("No categorical columns.")
