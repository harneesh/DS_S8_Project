import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from utils.analysis import top_correlations
from utils.data_loader import categorical_cols, datetime_cols, low_cardinality_cols, numeric_cols
from utils.plots import add_trendline, correlation_heatmap, sample_rows, show
from utils.state import require_data, show_flash

show_flash()
df = require_data()
st.title("🔗 Bivariate & Multivariate Analysis")
st.caption("Relationships between two or more columns.")

num = numeric_cols(df)
groups = low_cardinality_cols(df, 50)
NONE = "(none)"

tab_scatter, tab_corr, tab_numcat, tab_catcat, tab_pair, tab_time = st.tabs(
    ["Scatter plot", "Correlation", "Numeric vs category", "Category vs category", "Pair plot", "Trend over time"])

# ---------- scatter ----------
with tab_scatter:
    if len(num) < 2:
        st.info("Needs at least two numeric columns.")
    else:
        c = st.columns(4)
        x = c[0].selectbox("X axis", num, key="sc_x")
        y = c[1].selectbox("Y axis", num, index=1, key="sc_y")
        color = c[2].selectbox("Colour by", [NONE] + [col for col in df.columns if col not in (x, y)], key="sc_c")
        size = c[3].selectbox("Size by", [NONE] + [col for col in num if col not in (x, y)], key="sc_s")
        trend = st.checkbox("Show linear trend line", value=True)
        data = sample_rows(df)
        if size != NONE:  # marker size cannot be negative or missing
            data = data[data[size].notna()]
            data = data.assign(**{size: data[size] - min(data[size].min(), 0)})
        fig = px.scatter(data, x=x, y=y, color=None if color == NONE else color,
                         size=None if size == NONE else size, opacity=0.7,
                         title=f"{y} vs {x}" + (f" (random sample of {len(data):,} rows)" if len(data) < len(df) else ""))
        r = add_trendline(fig, df[x], df[y]) if trend else None
        show(fig)
        if r is not None:
            strength = "very weak" if abs(r) < 0.2 else "weak" if abs(r) < 0.4 else "moderate" if abs(r) < 0.6 \
                else "strong" if abs(r) < 0.8 else "very strong"
            st.info(f"Pearson correlation r = **{r:.3f}**: a {strength} "
                    f"{'positive' if r > 0 else 'negative'} linear relationship (R² = {r * r:.3f}).")

# ---------- correlation ----------
with tab_corr:
    if len(num) < 2:
        st.info("Needs at least two numeric columns.")
    else:
        c1, c2 = st.columns([1, 3])
        method = c1.radio("Method", ["pearson", "spearman", "kendall"],
                          help="Pearson = linear relationship; Spearman/Kendall = monotonic, based on ranks "
                               "(less sensitive to outliers).")
        cols = c2.multiselect("Columns", num, default=num[:15])
        if len(cols) >= 2:
            corr = df[cols].corr(method=method)
            show(correlation_heatmap(corr, f"{method.title()} correlation matrix"))
            st.markdown("##### Strongest relationships")
            st.dataframe(top_correlations(df[cols], method), hide_index=True, width="stretch")
        else:
            st.caption("Pick at least two columns.")

# ---------- numeric vs categorical ----------
with tab_numcat:
    if not num or not groups:
        st.info("Needs a numeric column and a categorical column with at most 50 unique values.")
    else:
        c = st.columns(3)
        y = c[0].selectbox("Numeric column", num, key="nc_y")
        g = c[1].selectbox("Group by", [col for col in groups if col != y], key="nc_g")
        kind = c[2].radio("Chart", ["Box", "Violin", "Mean bar"], horizontal=True)
        if kind == "Box":
            fig = px.box(df, x=g, y=y, color=g, points="outliers", title=f"{y} by {g}")
        elif kind == "Violin":
            fig = px.violin(df, x=g, y=y, color=g, box=True, title=f"{y} by {g}")
        else:
            agg = df.groupby(g, observed=True)[y].agg(["mean", "std", "count"]).reset_index()
            fig = px.bar(agg, x=g, y="mean", error_y="std", color=g, text_auto=".3s",
                         title=f"Mean {y} by {g} (error bar = 1 std dev)")
        show(fig.update_layout(showlegend=False))
        table = df.groupby(g, observed=True)[y].describe().round(3)
        st.dataframe(table, width="stretch")
        samples = [grp.dropna().values for _, grp in df.groupby(g, observed=True)[y] if grp.notna().sum() >= 2]
        if len(samples) >= 2:
            f, p = stats.f_oneway(*samples)
            verdict = "differ significantly" if p < 0.05 else "do not differ significantly"
            st.info(f"One-way ANOVA: F = {f:.3f}, p-value = {p:.4g}. The group means of **{y}** {verdict} "
                    f"across **{g}** at the 5% level.")

# ---------- categorical vs categorical ----------
with tab_catcat:
    cat_groups = [col for col in groups if col in categorical_cols(df) or df[col].nunique() <= 15]
    if len(cat_groups) < 2:
        st.info("Needs two categorical columns with at most 50 unique values.")
    else:
        c = st.columns(3)
        a = c[0].selectbox("Rows", cat_groups, key="cc_a")
        b = c[1].selectbox("Columns", [col for col in cat_groups if col != a], key="cc_b")
        norm = c[2].radio("Show", ["Counts", "Row %", "Column %"], horizontal=True)
        ct = pd.crosstab(df[a], df[b])
        shown = {"Counts": ct, "Row %": pd.crosstab(df[a], df[b], normalize="index") * 100,
                 "Column %": pd.crosstab(df[a], df[b], normalize="columns") * 100}[norm]
        show(px.imshow(shown.round(1), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                       title=f"{a} × {b} ({norm.lower()})"))
        show(px.histogram(df, x=a, color=b, barmode="group", title=f"{a} split by {b}"))
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            n = ct.values.sum()
            cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
            verdict = "are associated" if p < 0.05 else "look independent"
            st.info(f"Chi-square test: χ² = {chi2:.2f}, dof = {dof}, p-value = {p:.4g}, Cramér's V = {cramers_v:.3f}. "
                    f"**{a}** and **{b}** {verdict} at the 5% level.")

# ---------- pair plot ----------
with tab_pair:
    if len(num) < 2:
        st.info("Needs at least two numeric columns.")
    else:
        c1, c2 = st.columns([3, 1])
        cols = c1.multiselect("Columns (2–6)", num, default=num[:4], max_selections=6, key="pp_cols")
        color = c2.selectbox("Colour by", [NONE] + [col for col in groups if col not in cols], key="pp_c")
        if len(cols) >= 2:
            data = sample_rows(df, 2000)
            fig = px.scatter_matrix(data, dimensions=cols, color=None if color == NONE else color, opacity=0.6,
                                    title="Pair plot" + (f" (sample of {len(data):,} rows)" if len(data) < len(df) else ""))
            fig.update_traces(diagonal_visible=False, marker=dict(size=4))
            fig.update_layout(height=180 * len(cols) + 100)
            show(fig)

# ---------- trend over time ----------
with tab_time:
    dates = datetime_cols(df)
    if not dates or not num:
        st.info("Needs a date/time column and a numeric column. "
                "Convert a text column to date/time on the Data Quality page if needed.")
    else:
        c = st.columns(4)
        t = c[0].selectbox("Date column", dates)
        y = c[1].selectbox("Value", num, key="ts_y")
        agg = c[2].selectbox("Aggregate", ["mean", "sum", "median", "count", "min", "max"])
        freq_label = c[3].selectbox("Per", ["Day", "Week", "Month", "Quarter", "Year"], index=2)
        freq = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}[freq_label]
        g = st.selectbox("Separate lines by", [NONE] + [col for col in groups if df[col].nunique() <= 12], key="ts_g")
        keys = [df[t].dt.to_period(freq).dt.to_timestamp()] + ([df[g]] if g != NONE else [])
        series = df.groupby(keys, observed=True)[y].agg(agg).reset_index()
        show(px.line(series, x=t, y=y, color=None if g == NONE else g, markers=True,
                     title=f"{agg.title()} of {y} per {freq_label.lower()}"))
