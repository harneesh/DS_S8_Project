import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from utils.analysis import numeric_summary
from utils.data_loader import datetime_cols, low_cardinality_cols, numeric_cols
from utils.plots import qq_plot, show
from utils.state import require_data, show_flash

show_flash()
df = require_data()
st.title("📈 Univariate Analysis")
st.caption("Look at one column at a time: its distribution, centre, spread and shape.")

c1, c2 = st.columns([2, 1])
col = c1.selectbox("Column", list(df.columns))
s = df[col]
st.caption(f"Type `{s.dtype}` · {s.notna().sum():,} values · {s.isna().sum():,} missing · {s.nunique():,} unique")

# ---------- numeric ----------
if col in numeric_cols(df):
    group_opts = ["(none)"] + [c for c in low_cardinality_cols(df, 15) if c != col]
    color = c2.selectbox("Split by", group_opts)
    color = None if color == "(none)" else color

    summary = numeric_summary(s)
    m = st.columns(8)
    for box, (k, v) in zip(m, summary.items()):
        box.metric(k, f"{v:,.3g}" if isinstance(v, float) else f"{v:,}")

    tab_hist, tab_box, tab_qq, tab_ecdf = st.tabs(["Histogram", "Box & violin", "Normality (Q-Q)", "Cumulative (ECDF)"])
    with tab_hist:
        b1, b2 = st.columns([3, 1])
        bins = b1.slider("Number of bins", 5, 150, 30)
        hist_norm = b2.radio("Y axis", ["count", "percent"], horizontal=True)
        fig = px.histogram(df, x=col, color=color, nbins=bins, marginal="box", barmode="overlay",
                           histnorm=None if hist_norm == "count" else "percent", opacity=0.75,
                           title=f"Distribution of {col}")
        fig.add_vline(x=summary["Mean"], line_dash="dash", line_color="red", annotation_text="mean")
        fig.add_vline(x=summary["Median"], line_dash="dot", line_color="green", annotation_text="median")
        show(fig)
        skew = summary["Skewness"]
        shape = ("roughly symmetric" if abs(skew) < 0.5 else "moderately skewed" if abs(skew) < 1
                 else "highly skewed") + (" to the right" if skew > 0.5 else " to the left" if skew < -0.5 else "")
        st.info(f"The distribution of **{col}** is **{shape}** (skewness = {skew:.2f}).")
    with tab_box:
        l, r = st.columns(2)
        with l:
            show(px.box(df, y=col, x=color, color=color, points="outliers", title=f"Box plot of {col}"))
        with r:
            show(px.violin(df, y=col, x=color, color=color, box=True, title=f"Violin plot of {col}"))
    with tab_qq:
        clean = s.dropna()
        show(qq_plot(clean))
        if len(clean) >= 8:
            if len(clean) <= 5000:
                name, (stat, p) = "Shapiro-Wilk", stats.shapiro(clean)
            else:
                name, (stat, p) = "D'Agostino K²", stats.normaltest(clean)
            verdict = "looks **normally distributed**" if p > 0.05 else "is **not normally distributed**"
            st.info(f"{name} test: statistic = {stat:.4f}, p-value = {p:.4g}. At the 5% level, {col} {verdict}.")
    with tab_ecdf:
        show(px.ecdf(df, x=col, color=color, title=f"Cumulative distribution of {col}"))

# ---------- date/time ----------
elif col in datetime_cols(df):
    freq_label = c2.selectbox("Group by", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
    freq = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}[freq_label]
    m = st.columns(3)
    m[0].metric("Earliest", str(s.min().date()))
    m[1].metric("Latest", str(s.max().date()))
    m[2].metric("Span (days)", f"{(s.max() - s.min()).days:,}")
    counts = s.dropna().dt.to_period(freq).value_counts().sort_index()
    trend = pd.DataFrame({col: counts.index.to_timestamp(), "Count": counts.values})
    show(px.line(trend, x=col, y="Count", markers=True, title=f"Records per {freq_label.lower()}"))
    show(px.bar(s.dt.day_name().value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).fillna(0)
        .rename_axis("Weekday").reset_index(name="Count"), x="Weekday", y="Count", title="Records by weekday"))

# ---------- categorical ----------
else:
    nunique = s.nunique()
    top_n = c2.slider("Show top categories", 3, max(3, min(50, nunique)), min(15, max(3, nunique))) if nunique > 3 else nunique
    counts = s.astype("string").fillna("(missing)").value_counts()
    top = counts.head(top_n).rename_axis(col).reset_index(name="Count")
    if len(counts) > top_n:
        top = pd.concat([top, pd.DataFrame({col: ["(other)"], "Count": [counts.iloc[top_n:].sum()]})])
    top["Percent"] = (top["Count"] / top["Count"].sum() * 100).round(2)

    m = st.columns(4)
    m[0].metric("Unique values", f"{nunique:,}")
    m[1].metric("Most frequent", str(counts.index[0]))
    m[2].metric("Its share", f"{counts.iloc[0] / counts.sum():.1%}")
    m[3].metric("Missing", f"{s.isna().sum():,}")

    l, r = st.columns([3, 2])
    with l:
        show(px.bar(top, x=col, y="Count", text="Count", color=col, title=f"Value counts of {col}")
             .update_layout(showlegend=False))
    with r:
        show(px.pie(top, names=col, values="Count", hole=0.4, title="Share of each category"))
    st.dataframe(top, hide_index=True, width="stretch")
