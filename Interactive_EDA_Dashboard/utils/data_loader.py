"""Loading datasets (uploaded files or bundled samples) and detecting column types."""
import io
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SAMPLE_DATASETS = {
    "Employees (messy demo data)": lambda: pd.read_csv(DATA_DIR / "sample_employees.csv"),
    "Iris flowers": px.data.iris,
    "Restaurant tips": px.data.tips,
    "Gapminder (countries)": px.data.gapminder,
}


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Read CSV-like bytes, trying common encodings and sniffing the delimiter if needed."""
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            if df.shape[1] == 1:  # probably not comma-separated -> let pandas sniff ; \t | etc.
                df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
            return df
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error


def _parse_date_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns whose name suggests a date/time and that mostly parse as dates."""
    df = df.copy()
    hints = ("date", "time", "day", "month", "year_month", "timestamp")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if not any(h in str(col).lower() for h in hints):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        non_null = df[col].notna().sum()
        if non_null and parsed.notna().sum() / non_null >= 0.9:
            df[col] = parsed
    return df


@st.cache_data(show_spinner="Reading file...")
def load_uploaded(raw: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".json"):
        df = pd.read_json(io.BytesIO(raw))
    else:
        df = _read_csv(raw)
    df.columns = [str(c).strip() for c in df.columns]
    return _parse_date_like_columns(df)


@st.cache_data
def load_sample(name: str) -> pd.DataFrame:
    return _parse_date_like_columns(SAMPLE_DATASETS[name]())


# ---------- column type helpers ----------

def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def datetime_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def categorical_cols(df: pd.DataFrame) -> list[str]:
    skip = set(numeric_cols(df)) | set(datetime_cols(df))
    return [c for c in df.columns if c not in skip]


def low_cardinality_cols(df: pd.DataFrame, max_unique: int = 30) -> list[str]:
    """Columns usable for grouping/colouring: categorical (or small-integer) with few unique values."""
    return [c for c in df.columns
            if c not in datetime_cols(df) and 1 < df[c].nunique() <= max_unique]
