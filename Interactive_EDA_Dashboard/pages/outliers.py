import plotly.express as px
import streamlit as st

from utils.analysis import outlier_bounds, outlier_mask, outlier_table
from utils.data_loader import numeric_cols
from utils.plots import sample_rows, show
from utils.state import apply_change, require_data, show_flash

show_flash()
df = require_data()
ss = st.session_state
st.title("🎯 Outlier Detection")

num = [c for c in numeric_cols(df) if df[c].dropna().nunique() > 2]  # binary 0/1 columns have no outliers
if not num:
    st.info("No numeric columns (with more than two distinct values) to analyse.")
    st.stop()

c1, c2 = st.columns([1, 2])
method = c1.radio("Method", ["IQR", "Z-score"], horizontal=True,
                  help="IQR: outside [Q1 − k·IQR, Q3 + k·IQR]. Robust, no normality assumption.\n\n"
                       "Z-score: more than z standard deviations from the mean. Best for roughly normal data.")
if method == "IQR":
    param = c2.slider("k (IQR multiplier)", 1.0, 4.0, 1.5, 0.1, help="1.5 = standard, 3 = extreme outliers only")
else:
    param = c2.slider("z threshold", 1.5, 5.0, 3.0, 0.1)

st.subheader("Summary across all numeric columns")
table = outlier_table(df[num], method, param)
l, r = st.columns([3, 2])
l.dataframe(table, hide_index=True, width="stretch")
with r:
    show(px.bar(table.sort_values("Outliers %"), x="Outliers %", y="Column", orientation="h",
                text="Outliers", title="Share of outliers per column"))

st.subheader("Inspect one column")
col = st.selectbox("Column", num)
s = df[col]
lo, hi = outlier_bounds(s, method, param)
mask = outlier_mask(s, method, param)
m = st.columns(4)
m[0].metric("Outliers", f"{int(mask.sum()):,}")
m[1].metric("Share", f"{mask.sum() / s.notna().sum():.2%}")
m[2].metric("Lower limit", f"{lo:,.4g}")
m[3].metric("Upper limit", f"{hi:,.4g}")

plot_df = df.assign(_status=mask.map({True: "Outlier", False: "Normal"}), _row=df.index)
l, r = st.columns(2)
with l:
    show(px.box(df, y=col, points="outliers", title=f"Box plot of {col}"))
with r:
    data = sample_rows(plot_df)
    fig = px.scatter(data, x="_row", y=col, color="_status", labels={"_row": "Row index", "_status": ""},
                     color_discrete_map={"Normal": "#1f77b4", "Outlier": "#d62728"}, title=f"{col} by row")
    fig.add_hline(y=lo, line_dash="dash", line_color="gray", annotation_text="lower limit")
    fig.add_hline(y=hi, line_dash="dash", line_color="gray", annotation_text="upper limit")
    show(fig)

if mask.any():
    with st.expander(f"Show the {int(mask.sum())} outlier rows"):
        st.dataframe(df[mask], width="stretch")

    st.markdown("##### Treat outliers")
    st.caption("Applied to the full dataset, with limits computed from it. Undo from the sidebar.")
    full = ss.df[col]
    flo, fhi = outlier_bounds(full, method, param)
    full_mask = outlier_mask(full, method, param)
    b1, b2 = st.columns(2)
    if b1.button(f"Remove {int(full_mask.sum())} outlier rows", width="stretch"):
        apply_change(ss.df[~full_mask].reset_index(drop=True),
                     f"Removed {int(full_mask.sum())} outlier rows in {col} ({method}, {param})")
        st.rerun()
    if b2.button("Cap values at the limits (winsorize)", width="stretch"):
        new = ss.df.copy()
        new[col] = new[col].clip(flo, fhi)
        apply_change(new, f"Capped {col} to [{flo:.4g}, {fhi:.4g}] ({method}, {param})")
        st.rerun()
else:
    st.success(f"No outliers in {col} with these settings.")
