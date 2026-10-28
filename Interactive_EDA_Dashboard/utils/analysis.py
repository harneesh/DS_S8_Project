"""Statistical summaries, outlier detection and automatic insights."""
import numpy as np
import pandas as pd

from utils.data_loader import categorical_cols, numeric_cols


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        rows.append({
            "Column": c,
            "Type": str(s.dtype),
            "Non-null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(s.isna().mean() * 100, 2),
            "Unique": int(s.nunique()),
            "Example": str(non_null.iloc[0]) if len(non_null) else "",
        })
    return pd.DataFrame(rows)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    out = pd.DataFrame({"Column": miss.index, "Missing": miss.values,
                        "Missing %": (miss.values / max(len(df), 1) * 100).round(2)})
    return out[out["Missing"] > 0].sort_values("Missing", ascending=False).reset_index(drop=True)


def numeric_summary(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "Count": len(s), "Mean": s.mean(), "Median": s.median(), "Std dev": s.std(),
        "Min": s.min(), "Max": s.max(), "Skewness": s.skew(), "Kurtosis": s.kurt(),
    }


# ---------- outliers ----------

def outlier_bounds(s: pd.Series, method: str = "IQR", param: float = 1.5) -> tuple[float, float]:
    """Lower/upper limits outside which values count as outliers.

    IQR:     [Q1 - k*IQR, Q3 + k*IQR]
    Z-score: [mean - z*std, mean + z*std]
    """
    s = s.dropna()
    if method == "IQR":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - param * iqr, q3 + param * iqr
    mean, std = s.mean(), s.std()
    return mean - param * std, mean + param * std


def outlier_mask(s: pd.Series, method: str = "IQR", param: float = 1.5) -> pd.Series:
    lo, hi = outlier_bounds(s, method, param)
    return (s < lo) | (s > hi)


def outlier_table(df: pd.DataFrame, method: str = "IQR", param: float = 1.5) -> pd.DataFrame:
    rows = []
    for c in numeric_cols(df):
        s = df[c]
        if s.dropna().nunique() < 3:  # skip constant and binary (0/1) columns
            continue
        lo, hi = outlier_bounds(s, method, param)
        n = int(outlier_mask(s, method, param).sum())
        rows.append({"Column": c, "Outliers": n, "Outliers %": round(n / s.notna().sum() * 100, 2),
                     "Lower limit": round(lo, 4), "Upper limit": round(hi, 4)})
    return pd.DataFrame(rows)


# ---------- correlations ----------

def top_correlations(df: pd.DataFrame, method: str = "pearson", n: int = 10) -> pd.DataFrame:
    corr = df[numeric_cols(df)].corr(method=method)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["Column A", "Column B", "Correlation"]
    pairs["Strength"] = pairs["Correlation"].abs()
    pairs = pairs.sort_values("Strength", ascending=False).head(n)
    pairs["Correlation"] = pairs["Correlation"].round(3)
    return pairs.drop(columns="Strength").reset_index(drop=True)


# ---------- automatic insights ----------

def generate_insights(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Rule-based observations about the dataset as (level, message); level is warning/info."""
    insights = []
    n = len(df)
    if n == 0:
        return insights

    dups = int(df.duplicated().sum())
    if dups:
        insights.append(("warning", f"**{dups}** duplicate rows ({dups / n:.1%} of the data)."))

    for c, pct in (df.isna().mean() * 100).items():
        if pct >= 40:
            insights.append(("warning", f"**{c}** is {pct:.1f}% missing; consider dropping it."))
        elif pct >= 5:
            insights.append(("info", f"**{c}** has {pct:.1f}% missing values."))

    for c in df.columns:
        nunique = df[c].nunique()
        if nunique <= 1:
            insights.append(("warning", f"**{c}** has a single constant value, so it carries no information."))
        elif c in categorical_cols(df) and nunique == df[c].notna().sum() and n > 20:
            insights.append(("info", f"**{c}** is unique for every row, so it is probably an ID column."))
        elif c in categorical_cols(df) and nunique <= 20:
            top_share = df[c].value_counts(normalize=True).iloc[0]
            if top_share >= 0.8:
                insights.append(("info", f"**{c}** is imbalanced: one value covers {top_share:.0%} of rows."))

    for c in numeric_cols(df):
        s = df[c].dropna()
        if s.nunique() < 3:
            continue
        skew = s.skew()
        if abs(skew) >= 1:
            side = "right (long tail of large values)" if skew > 0 else "left (long tail of small values)"
            insights.append(("info", f"**{c}** is strongly skewed to the {side}, skewness = {skew:.2f}."))
        pct = outlier_mask(s).mean()
        if pct >= 0.05:
            insights.append(("info", f"**{c}** has {pct:.1%} outliers by the IQR rule."))

    if len(numeric_cols(df)) >= 2:
        for _, row in top_correlations(df, n=5).iterrows():
            if abs(row["Correlation"]) >= 0.7:
                direction = "positive" if row["Correlation"] > 0 else "negative"
                insights.append(("info", f"Strong {direction} correlation between **{row['Column A']}** "
                                         f"and **{row['Column B']}** (r = {row['Correlation']:.2f})."))
    return insights
