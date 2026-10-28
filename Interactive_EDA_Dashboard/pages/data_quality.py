import pandas as pd
import plotly.express as px
import streamlit as st

from utils.analysis import missing_summary
from utils.data_loader import numeric_cols
from utils.plots import missing_heatmap, show
from utils.state import apply_change, require_data, show_flash

show_flash()
require_data()
ss = st.session_state
df = ss.df  # cleaning always works on the full dataset, not the filtered view
st.title("🧹 Data Quality & Cleaning")
if len(ss.df_view) != len(df):
    st.caption("Note: this page works on the **full** dataset; sidebar filters are ignored here.")

tab_missing, tab_dups, tab_clean, tab_export = st.tabs(
    ["Missing values", "Duplicates", "Clean data", "History & export"])

# ---------- missing values ----------
with tab_missing:
    miss = missing_summary(df)
    c1, c2 = st.columns(2)
    c1.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c2.metric("Rows with any missing value", f"{int(df.isna().any(axis=1).sum()):,}")
    if miss.empty:
        st.success("No missing values 🎉")
    else:
        left, right = st.columns([1, 2])
        left.dataframe(miss, hide_index=True, width="stretch")
        with right:
            show(px.bar(miss, x="Column", y="Missing %", text="Missing %",
                        title="Missing values per column (%)"))
        show(missing_heatmap(df))

# ---------- duplicates ----------
with tab_dups:
    subset = st.multiselect("Check duplicates using these columns (empty = all columns)", list(df.columns))
    dup_mask = df.duplicated(subset=subset or None, keep=False)
    n_dups = int(df.duplicated(subset=subset or None).sum())
    st.metric("Duplicate rows (extra copies)", f"{n_dups:,}")
    if n_dups:
        st.dataframe(df[dup_mask].sort_values(subset or list(df.columns)).head(200), width="stretch")
        if st.button("Remove duplicates (keep first copy)", type="primary"):
            apply_change(df.drop_duplicates(subset=subset or None).reset_index(drop=True),
                         f"Removed {n_dups} duplicate rows" + (f" (by {', '.join(subset)})" if subset else ""))
            st.rerun()
    else:
        st.success("No duplicate rows.")

# ---------- cleaning tools ----------
with tab_clean:
    st.markdown("Each action changes the working dataset. Use **Undo** or **Reset** in the sidebar to revert.")

    with st.container(border=True):
        st.markdown("##### Handle missing values")
        cols_with_missing = miss["Column"].tolist()
        c1, c2, c3 = st.columns([2, 1.3, 1])
        cols = c1.multiselect("Columns", list(df.columns), default=cols_with_missing, key="fill_cols")
        strategy = c2.selectbox("Strategy", ["Drop rows", "Fill with mean", "Fill with median",
                                             "Fill with mode (most frequent)", "Fill with a constant",
                                             "Forward fill", "Backward fill"])
        const = c3.text_input("Constant", "0", disabled=strategy != "Fill with a constant")
        if st.button("Apply", key="apply_fill", disabled=not cols):
            new = df.copy()
            skipped = []
            if strategy == "Drop rows":
                new = new.dropna(subset=cols).reset_index(drop=True)
            for c in cols if strategy != "Drop rows" else []:
                if strategy in ("Fill with mean", "Fill with median"):
                    if c not in numeric_cols(new):
                        skipped.append(c)
                        continue
                    new[c] = new[c].fillna(new[c].mean() if "mean" in strategy else new[c].median())
                elif strategy.startswith("Fill with mode"):
                    mode = new[c].mode()
                    if not mode.empty:
                        new[c] = new[c].fillna(mode.iloc[0])
                elif strategy == "Fill with a constant":
                    value = pd.to_numeric(const, errors="coerce") if c in numeric_cols(new) else const
                    new[c] = new[c].fillna(const if pd.isna(value) else value)
                elif strategy == "Forward fill":
                    new[c] = new[c].ffill()
                else:
                    new[c] = new[c].bfill()
            done = [c for c in cols if c not in skipped]
            if skipped:
                st.warning(f"Skipped non-numeric columns for {strategy.lower()}: {', '.join(skipped)}")
            if done:
                apply_change(new, f"{strategy} on {', '.join(done)}")
                st.rerun()

    with st.container(border=True):
        st.markdown("##### Change column type")
        c1, c2 = st.columns(2)
        col = c1.selectbox("Column", list(df.columns), key="type_col")
        target = c2.selectbox("Convert to", ["Number", "Date/time", "Category", "Text"])
        st.caption(f"Current type: `{df[col].dtype}`")
        if st.button("Convert", key="convert"):
            new = df.copy()
            before = new[col].isna().sum()
            if target == "Number":
                new[col] = pd.to_numeric(new[col], errors="coerce")
            elif target == "Date/time":
                new[col] = pd.to_datetime(new[col], errors="coerce", format="mixed")
            elif target == "Category":
                new[col] = new[col].astype("category")
            else:
                new[col] = new[col].astype("string")
            lost = int(new[col].isna().sum() - before)
            apply_change(new, f"Converted {col} to {target.lower()}" + (f" ({lost} values became missing)" if lost else ""))
            st.rerun()

    with st.container(border=True):
        st.markdown("##### Drop or rename columns")
        c1, c2 = st.columns(2)
        with c1:
            drop = st.multiselect("Columns to drop", list(df.columns), key="drop_cols")
            if st.button("Drop columns", disabled=not drop):
                apply_change(df.drop(columns=drop), f"Dropped columns: {', '.join(drop)}")
                st.rerun()
        with c2:
            old = st.selectbox("Column to rename", list(df.columns), key="rename_col")
            new_name = st.text_input("New name", key="rename_to")
            if st.button("Rename", disabled=not new_name.strip()):
                if new_name.strip() in df.columns:
                    st.error("A column with that name already exists.")
                else:
                    apply_change(df.rename(columns={old: new_name.strip()}), f"Renamed {old} → {new_name.strip()}")
                    st.rerun()

    with st.container(border=True):
        st.markdown("##### Clean text columns")
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        c1, c2 = st.columns([2, 1])
        tcols = c1.multiselect("Text columns", text_cols, key="text_cols")
        op = c2.selectbox("Operation", ["Trim spaces", "lowercase", "Title Case", "UPPERCASE"])
        if st.button("Apply", key="apply_text", disabled=not tcols):
            new = df.copy()
            for c in tcols:
                s = new[c].astype("string")
                new[c] = {"Trim spaces": s.str.strip(), "lowercase": s.str.lower(),
                          "Title Case": s.str.title(), "UPPERCASE": s.str.upper()}[op]
            apply_change(new, f"{op} on {', '.join(tcols)}")
            st.rerun()

# ---------- history and export ----------
with tab_export:
    c1, c2 = st.columns(2)
    c1.metric("Original shape", f"{ss.df_original.shape[0]:,} × {ss.df_original.shape[1]}")
    c2.metric("Current shape", f"{df.shape[0]:,} × {df.shape[1]}",
              delta=f"{df.shape[0] - ss.df_original.shape[0]:,} rows")
    st.markdown("##### Cleaning steps applied")
    if ss.log:
        for i, step in enumerate(ss.log, 1):
            st.markdown(f"{i}. {step}")
    else:
        st.caption("No cleaning steps yet.")

    st.markdown("##### Download")
    base = str(ss.dataset_name).rsplit(".", 1)[0].replace(" ", "_")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Cleaned dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}_cleaned.csv", mime="text/csv", width="stretch")
    d2.download_button("⬇ Filtered view (CSV)", ss.df_view.to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}_filtered.csv", mime="text/csv", width="stretch")
    st.caption("Tip: every chart can be saved as PNG with the 📷 icon in its top-right toolbar.")
