"""Reusable Plotly charts. Every chart has a camera icon in its toolbar to download it as PNG."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

MAX_PLOT_ROWS = 5000  # sample larger datasets for heavy point-based charts


def show(fig: go.Figure) -> None:
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def sample_rows(df: pd.DataFrame, n: int = MAX_PLOT_ROWS) -> pd.DataFrame:
    return df.sample(n, random_state=0) if len(df) > n else df


def missing_heatmap(df: pd.DataFrame, max_rows: int = 1000) -> go.Figure:
    data = df.head(max_rows).isna().astype(int).T
    fig = px.imshow(data, color_continuous_scale=[[0, "#e8eef7"], [1, "#d62728"]], aspect="auto",
                    labels=dict(x="Row", y="Column", color="Missing"),
                    title=f"Missing values map (first {min(len(df), max_rows)} rows; red = missing)")
    fig.update_coloraxes(showscale=False)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", title=title)
    fig.update_layout(height=max(400, 45 * len(corr)))
    return fig


def qq_plot(s: pd.Series) -> go.Figure:
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(s.dropna(), dist="norm")
    fig = go.Figure()
    fig.add_scatter(x=theoretical, y=ordered, mode="markers", name="Data", marker=dict(size=5))
    fig.add_scatter(x=theoretical, y=slope * theoretical + intercept, mode="lines",
                    name="Normal reference", line=dict(color="red"))
    fig.update_layout(title=f"Q-Q plot of {s.name} (points on the line = normally distributed)",
                      xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
    return fig


def add_trendline(fig: go.Figure, x: pd.Series, y: pd.Series) -> float | None:
    """Add a least-squares line to a scatter plot and return Pearson r."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2:
        return None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                    line=dict(color="black", dash="dash"))
    return float(np.corrcoef(x[mask], y[mask])[0, 1])
