
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hypersonic EDA — notebook generator
-----------------------------------
Creates a structured .ipynb with:
  - Title/Overview
  - Reproducible data-loading cell
  - Helper plotting functions (categorical & numeric)
  - For each categorical feature:  (1) plot cell, (2) table cell (value_counts)
  - For each numeric feature:      (1) plot cell, (2) table cell (summary stats)
  - Target analysis (only when --target is provided; supports multiple targets)

Usage:
  python hy.py --input INPUT [--target "col1, col2; col3"] [--table TABLE] [--max-cat 30] [--max-num 30] [--output eda.ipynb]
"""

import argparse
import io
import os
import re
import sys
import textwrap
from datetime import datetime
from typing import List, Optional

import nbformat as nbf
import pandas as pd


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _is_url(path: str) -> bool:
    return path.lower().startswith(("http://", "https://"))

def _esc(s: str) -> str:
    # Escape triple quotes inside f-strings for safe embedding in code cells
    return s.replace('"""', r'\"\"\"')

def infer_feature_types(df: pd.DataFrame, max_unique_for_categorical: int = 30):
    """Split columns into categorical vs numeric based on dtype & cardinality."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Object/string/boolean and low-cardinality numerics treated as categorical
    candidate_cats = df.columns.difference(numeric_cols).tolist()
    # Add numerics with low cardinality (useful for IDs with few buckets)
    for col in numeric_cols:
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= max_unique_for_categorical:
            candidate_cats.append(col)
    # Deduplicate while preserving order
    seen = set()
    categorical_cols = []
    for c in candidate_cats:
        if c not in seen:
            seen.add(c)
            categorical_cols.append(c)
    return categorical_cols, numeric_cols

def _parse_targets(arg: Optional[str]) -> List[str]:
    """Split --target on comma/semicolon, trim, drop empties, preserve order, dedupe."""
    if not arg:
        return []
    parts = re.split(r'[;,]', arg)
    seen, out = set(), []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ------------------------------------------------------------
# Notebook cell helpers
# ------------------------------------------------------------

def add_md(nb, text: str):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(nb, code: str, hide_input: bool = False):
    cell = nbf.v4.new_code_cell(code)
    if hide_input:
        cell.metadata = {"tags": ["hide-input"]}
    nb.cells.append(cell)

def add_section(nb, title: str, anchor: Optional[str] = None):
    if anchor:
        add_md(nb, f"## <a id='{anchor}'></a>{title}")
    else:
        add_md(nb, f"## {title}")


# ------------------------------------------------------------
# Notebook skeleton + EDA cells
# ------------------------------------------------------------

def build_notebook(
    input_source: str,
    output_path: str,
    targets: List[str],
    table: Optional[str],
    features_categorical: List[str],
    features_numeric: List[str]
):
    nb = nbf.v4.new_notebook()

    # ---------------- Title & Overview ----------------
    title = "# Hypersonic EDA Report"
    toc = textwrap.dedent(
        """
        - [Overview](#overview)
        - [Data Loading](#data-loading)
        - [Helper Functions](#helpers)
        - [Categorical Features](#categorical)
        - [Numeric Features](#numeric)
        - [Target analysis](#target-analysis)
        - [Notes](#notes)
        """
    )
    targets_label = ", ".join(targets) if targets else "—"
    meta = textwrap.dedent(
        f"""
        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        **Source:** `{_esc(input_source)}`
        **Target (if any):** `{_esc(targets_label)}`
        **Table (SQLite .db):** `{_esc(table or 'auto')}`
        """
    )
    add_md(nb, title)
    add_md(nb, toc)
    add_section(nb, "Overview", anchor="overview")
    add_md(nb, meta)

    # ---------------- Data Loading Cell ----------------
    add_section(nb, "Data Loading", anchor="data-loading")
    loader_code = f'''\
# This cell loads your dataset into a pandas DataFrame named `df`.
# It supports:
#  - CSV / Parquet (local path or HTTP URL)
#  - SQLite .db (local path or HTTP URL). For HTTP, the file is downloaded to a temp file.
# If your DB has multiple tables, set TABLE explicitly below.

DATA_SOURCE = r"""{_esc(input_source)}"""
TABLE = r"""{_esc(table or "")}""" or None

import os, io, tempfile, sqlite3
from contextlib import closing
import pandas as pd

def _is_url(path: str) -> bool:
    return path.lower().startswith(("http://", "https://"))

def _download_to_temp(url: str, suffix: str):
    import requests, tempfile
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return tmp_path

def load_data(source: str, table: str | None = None) -> pd.DataFrame:
    lower = source.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(source)
    if lower.endswith(".parquet"):
        return pd.read_parquet(source)
    if lower.endswith(".db") or lower.endswith(".sqlite"):
        db_path = source
        if _is_url(source):
            db_path = _download_to_temp(source, suffix=".db")
        with sqlite3.connect(db_path) as conn:
            # if no table specified, pick the first table in sqlite_master
            if table is None:
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
                    conn
                )
                if tables.empty:
                    raise RuntimeError("No tables found in SQLite DB.")
                table_name = tables.iloc[0, 0]
            else:
                table_name = table
            # NOTE: quoting the table name inside the *inner* f-string.
            return pd.read_sql_query(f'SELECT * FROM "{{table_name}}"', conn)

    # Fallback: try pandas (CSV)
    try:
        return pd.read_csv(source)
    except Exception:
        pass
    # The braces below are doubled so they survive to the notebook cell.
    raise ValueError(f"Unsupported source format: {{source}}")

df = load_data(DATA_SOURCE, TABLE)
print("Shape:", df.shape)
df.head()
'''
    add_code(nb, loader_code)

    # ---------------- Stats + Cleaning + Typo-check guidance + Unique values ----------------
    add_code(
        nb,
        textwrap.dedent(
            """
            # DataFrame statistics (includes non-numeric)
            import pandas as pd

            def stst(frame: pd.DataFrame) -> pd.DataFrame:
                \"\"\"Full-frame stats including non-numeric and datetimes.\"\"\"
                try:
                    return frame.describe(include="all", datetime_is_numeric=True)
                except TypeError:
                    # Older pandas fallback
                    return frame.describe(include="all")

            try:
                from IPython.display import display
                display(stst(df).T)
            except Exception:
                print(stst(df).T.to_string())
            """
        ).strip()
    )

    add_code(
        nb,
        textwrap.dedent(
            """
            # Basic cleaning of strings to remove tyo errors 
            # I normally dont strip spaces and underscores between words, but theres inconsistencies that make it easier just to do so
            for col in df.select_dtypes(include=['object', 'string']):
                df[col] = (
                    df[col]
                    .str.lower()                            # Converts text to lowercase
                    .str.strip()                            # Strip leading/trailing whitespace
                    .str.replace(r'\\s+', '', regex=True)   # Remove all whitespace (including between words)
                    .str.replace('_', '', regex=False)      # Remove all underscores
                    .str.replace('.', '', regex=False)      # Remove all periods
                )
            """
        ).strip()
    )

    add_md(nb, "### Carefully check and correct the Typo errors in Catagory Text.")

    if features_categorical:
        for _feat in features_categorical:
            add_code(
                nb,
                textwrap.dedent(
                    f"""
                    # Unique values for categorical feature: {_esc(_feat)}
                    _ser = df[{_esc(_feat)!r}]
                    try:
                        _ser = _ser.astype("string")
                    except Exception:
                        _ser = _ser.astype(str)
                    _ser = _ser.str.strip()

                    print("Feature:", {_esc(_feat)!r})
                    print("uv_count:", _ser.nunique(dropna=True))
                    _vals = sorted([v for v in _ser.dropna().unique() if str(v).strip().lower() != "nan"])
                    print(_vals)

                    del _ser, _vals
                    """
                ).strip()
            )
    else:
        add_md(nb, "_No categorical text features detected for unique-value listing._")

    # ---------------- Helper Functions ----------------
    add_section(nb, "Helper Functions", anchor="helpers")
    helpers = r'''\
# Plotting & table helpers — one figure per call, followed by table cells elsewhere.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SAFE_NA = "<NA>"  # ASCII placeholder to avoid glyph warnings
MAROON = "maroon"

def _safe_title(s):
    try:
        return str(s)
    except Exception:
        return "feature"

def plot_categorical(df: pd.DataFrame, feature_name: str, top_n: int = 30):
    """Bar chart of value counts (top_n)."""
    if feature_name not in df.columns:
        print(f"[warn] {feature_name} not in dataframe.")
        return
    s = df[feature_name]
    s_plot = s.astype(str).where(~pd.isna(s), other=SAFE_NA)
    counts_show = s_plot.value_counts(dropna=False).head(top_n)

    plt.figure(figsize=(10, 4))
    ax = counts_show.iloc[::-1].plot(kind="barh")
    ax.set_title(f"Categorical: {_safe_title(feature_name)} (top {min(top_n, len(counts_show))})")
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.show()

def categorical_table(df: pd.DataFrame, feature_name: str, normalize: bool = True, top_n: int = 50) -> pd.DataFrame:
    """Value counts table with counts & percents."""
    if feature_name not in df.columns:
        return pd.DataFrame()
    s = df[feature_name].astype("object")
    series = s.where(~pd.isna(s), other=SAFE_NA)
    counts = series.value_counts(dropna=False)
    perc = series.value_counts(dropna=False, normalize=True) * 100.0
    out = pd.DataFrame({"count": counts, "percent": perc}).reset_index()
    out.columns = [feature_name, "count", "percent"]
    out["percent"] = out["percent"].round(2)
    return out.head(top_n)

def plot_numeric(df: pd.DataFrame, feature_name: str, bins="auto"):
    """Histogram for numeric columns, with mean/median reference lines."""
    if feature_name not in df.columns:
        print(f"[warn] {feature_name} not in dataframe.")
        return
    s = pd.to_numeric(df[feature_name], errors="coerce").dropna()
    if s.empty:
        print(f"[warn] {feature_name} has no numeric data after coercion.")
        return
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.hist(s, bins=bins)
    ax.set_title(f"Numeric: {_safe_title(feature_name)}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frequency")
    mean_v, med_v = float(s.mean()), float(s.median())
    ax.axvline(mean_v, linestyle="--", linewidth=1)
    ax.axvline(med_v, linestyle=":", linewidth=1)
    ax.legend([f"mean={mean_v:.3g}", f"median={med_v:.3g}"])
    plt.tight_layout()
    plt.show()

def numeric_table(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Summary statistics + simple outlier markers (IQR method)."""
    if feature_name not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[feature_name], errors="coerce")
    desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame().T
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    out = desc.assign(
        iqr=iqr,
        lower_fence=lo,
        upper_fence=hi,
        n_outliers=((s < lo) | (s > hi)).sum()
    )
    return out

# ---------- Target-aware helpers (NEW) ----------

def plot_target_categorical(df: pd.DataFrame, target_col: str, top_n: int = 30):
    """Plot distribution of a categorical target (maroon)."""
    if target_col not in df.columns:
        print(f"[warn] target {target_col} not in dataframe.")
        return
    s = df[target_col].astype("object").where(~pd.isna(df[target_col]), other=SAFE_NA)
    counts = s.value_counts(dropna=False).head(top_n)
    plt.figure(figsize=(10, 4))
    ax = counts.plot(kind="bar", color=MAROON)
    ax.set_title(f"Target distribution: {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

def target_categorical_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Counts and % for a categorical target."""
    s = df[target_col].astype("object").where(~pd.isna(df[target_col]), other=SAFE_NA)
    vc = s.value_counts(dropna=False)
    pct = s.value_counts(dropna=False, normalize=True) * 100.0
    out = pd.DataFrame({"count": vc, "percent": pct}).reset_index()
    out.columns = [target_col, "count", "percent"]
    out["percent"] = out["percent"].round(2)
    return out

def plot_target_numeric(df: pd.DataFrame, target_col: str, bins="auto"):
    """Histogram of numeric target (maroon)."""
    s = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if s.empty:
        print(f"[warn] target {target_col} has no numeric data after coercion.")
        return
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.hist(s, bins=bins, color=MAROON)
    ax.set_title(f"Target distribution: {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def target_numeric_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Summary stats for a numeric target."""
    s = pd.to_numeric(df[target_col], errors="coerce")
    return s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame().T
'''
    add_code(nb, helpers, hide_input=False)

    # ---------------- Categorical Features ----------------
    add_section(nb, "Categorical Features", anchor="categorical")

    if not features_categorical:
        add_md(nb, "_No categorical features selected._")
    else:
        for feat in features_categorical:
            add_md(nb, f"### Categorical: `{_esc(feat)}` — Plot")
            add_code(nb, f"plot_categorical(df, '{_esc(feat)}', top_n=30)")
            add_md(nb, f"**Value Counts for `{_esc(feat)}`**")
            add_code(
                nb,
                textwrap.dedent(
                    f"""\
                    _tbl = categorical_table(df, '{_esc(feat)}', normalize=True, top_n=50)
                    try:
                        from IPython.display import display
                        display(_tbl)
                    except Exception:
                        print(_tbl.to_string(index=False))
                    del _tbl
                    """
                )
            )

    # ---------------- Numeric Features ----------------
    add_section(nb, "Numeric Features", anchor="numeric")

    if not features_numeric:
        add_md(nb, "_No numeric features selected._")
    else:
        for feat in features_numeric:
            add_md(nb, f"### Numeric: `{_esc(feat)}` — Histogram")
            add_code(nb, f"plot_numeric(df, '{_esc(feat)}', bins='auto')")
            add_md(nb, f"**Summary for `{_esc(feat)}`**")
            add_code(
                nb,
                textwrap.dedent(
                    f"""\
                    _ntbl = numeric_table(df, '{_esc(feat)}')
                    try:
                        from IPython.display import display
                        display(_ntbl)
                    except Exception:
                        print(_ntbl.to_string(index=False))
                    del _ntbl
                    """
                )
            )

    # ---------------- Target analysis (supports multiple targets) ----------------
    add_section(nb, "Target analysis", anchor="target-analysis")
    add_md(nb, "### Target analysis.")

    # A) Target(s) summary cell — detects type per target, plots once, shows table
    add_code(
        nb,
        textwrap.dedent(
            f"""\
            TARGET_COLS = {targets!r}  # list of target column names
            if TARGET_COLS:
                import pandas as pd
                from IPython.display import display
                TC_numerical = []
                TC_catagorical = []

                for TARGET_COL in TARGET_COLS:
                    if TARGET_COL not in df.columns:
                        print("Target '" + str(TARGET_COL) + "' not found in dataframe; skipping.")
                        continue
                    if pd.api.types.is_numeric_dtype(df[TARGET_COL]):
                        TC_numerical.append(TARGET_COL)
                        # Numeric target — plot + stats table
                        plot_target_numeric(df, TARGET_COL, bins="auto")
                        _ttbl = target_numeric_table(df, TARGET_COL)
                        try:
                            display(_ttbl)
                        except Exception:
                            print(_ttbl.to_string(index=False))
                        del _ttbl

                        # Outlier report (IQR) for numeric target (NO PLOTS)
                        _s = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
                        if not _s.empty:
                            _q1, _q3 = _s.quantile(0.25), _s.quantile(0.75)
                            _iqr = _q3 - _q1
                            _lo = _q1 - 1.5 * _iqr
                            _hi = _q3 + 1.5 * _iqr
                            _n = _s.size
                            _nout = int(((_s < _lo) | (_s > _hi)).sum())
                            _pct = round((_nout / _n) * 100.0, 2) if _n else 0.0
                            _orep = pd.DataFrame({{
                                "q1": [_q1], "q3": [_q3], "iqr": [_iqr],
                                "lower_fence": [_lo], "upper_fence": [_hi],
                                "n": [_n], "n_outliers": [_nout], "pct_outliers": [_pct]
                            }})
                            try:
                                display(_orep)
                            except Exception:
                                print(_orep.to_string(index=False))
                            _rec = "Consider capping/winsorizing at the IQR fences" if _pct >= 1.0 else "Outliers are minimal; usually safe to keep."
                            _skew = float(_s.skew()) if _n else 0.0
                            if abs(_skew) > 1.0:
                                _rec += "; distribution is skewed (skew=" + ("%.2f" % _skew) + "), you may also try a log/Box-Cox transform."
                            print("Outlayers to be handled...... " + _rec)
                            del _s, _q1, _q3, _iqr, _lo, _hi, _n, _nout, _pct, _orep, _rec, _skew
                        else:
                            print("No numeric data available for outlier check on target.")

                    else:
                        TC_catagorical.append(TARGET_COL)
                        # Categorical target — plot + counts/percent table
                        plot_target_categorical(df, TARGET_COL, top_n=30)
                        _ctbl = target_categorical_table(df, TARGET_COL)
                        try:
                            display(_ctbl)
                        except Exception:
                            print(_ctbl.to_string(index=False))
                        del _ctbl

                print("TC_numerical:", TC_numerical)
                print("TC_catagorical:", TC_catagorical)
            else:
                print("No target specified (or not found).")
            """
        ).strip()
    )

    # B) Target imbalance report (NO PLOTS) — only for categorical targets
    add_code(
        nb,
        textwrap.dedent(
            f"""\
            # Target imbalance report: compute % distribution and flag minorities (<25%)
            TARGET_COLS = {targets!r}
            if TARGET_COLS:
                import pandas as pd
                from IPython.display import display
                minor_threshold = 25.0
                for TARGET_COL in TARGET_COLS:
                    if TARGET_COL not in df.columns:
                        continue
                    if pd.api.types.is_numeric_dtype(df[TARGET_COL]):
                        print("Imbalance check is only for categorical targets; numeric target '" + str(TARGET_COL) + "' detected — skipping.")
                        continue

                    s = df[TARGET_COL].astype("object").where(~pd.isna(df[TARGET_COL]), other=SAFE_NA)
                    counts = s.value_counts(dropna=False)
                    perc = (counts / counts.sum() * 100.0).round(2)
                    rep = pd.DataFrame({{"count": counts, "percent": perc}})
                    rep.index.name = TARGET_COL
                    rep["minority_flag"] = rep["percent"] < minor_threshold
                    try:
                        display(rep)
                    except Exception:
                        print(rep.to_string())

                    mins = rep[rep["minority_flag"]]
                    for idx, row in mins.iterrows():
                        print("Minority class in target " + str(TARGET_COL) + " -> '" + str(idx) + "': " + str(row['percent']) + "% — consider using SMOTE to balance.")
                    del s, counts, perc, rep, mins
            else:
                print("No target specified (or not found).")
            """
        ).strip()
    )

    # ---------------- Duplicate Rows Check ----------------
    add_section(nb, "Duplicate Rows", anchor="duplicates")
    add_code(
        nb,
        textwrap.dedent(
            """
            # Duplicate rows check (does not modify df)
            import pandas as pd
            orig_shape = df.shape
            dup_count = int(df.duplicated(keep='first').sum())
            if dup_count == 0:
                print("No duplicate records found.")
                print("Current shape:", orig_shape)
            else:
                print(f"{dup_count} - duplicate / identical rows, to be deleted .....to avoid data leakage and biased training/validation splits in modelling.")
                print("Current shape:", orig_shape)
                preview_shape = (orig_shape[0] - dup_count, orig_shape[1])
                print("Shape after removing duplicates (preview):", preview_shape)
            """
        ).strip()
    )

    # ---------------- Notes ----------------
    add_section(nb, "Notes", anchor="notes")
    add_md(nb, textwrap.dedent(
        """
        - Replace or extend the loader in **Data Loading** as needed (auth, formats, etc.).
        - Add domain context to interpret the numeric summary & categorical distributions.
        - Consider transformations (log, winsorization) for heavy-tailed numeric features.
        - For very long-tail categoricals, consider grouping infrequent classes as "Other".
        """
    ))

    # Closing action-reminder cell
    add_md(nb, "### Steps to do: Nan & empty cells imputation and further feature engineering to be done.")

    # ---------------- Write Notebook ----------------
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook written to: {output_path}")


# ------------------------------------------------------------
# Main (CLI)
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hypersonic EDA notebook builder")
    parser.add_argument("--input", required=True, help="CSV/Parquet/SQLite .db (path or URL)")
    parser.add_argument("--table", default=None, help="Table name if input is SQLite DB (optional)")
    parser.add_argument("--target", default=None, help="Target column(s) — separate multiple with comma or semicolon")
    parser.add_argument("--max-cat", type=int, default=30, help="Max categorical features to include")
    parser.add_argument("--max-num", type=int, default=30, help="Max numeric features to include")
    parser.add_argument("--output", default="eda.ipynb", help="Output notebook path (.ipynb)")
    args = parser.parse_args()

    targets = _parse_targets(args.target)

    # Light preview load to infer types & pick feature lists
    def load_preview(source: str, table: Optional[str]) -> pd.DataFrame:
        lower = source.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(source, nrows=50000)
        if lower.endswith(".parquet"):
            return pd.read_parquet(source)
        if lower.endswith(".db") or lower.endswith(".sqlite"):
            db_path = source
            if _is_url(source):
                try:
                    import requests, tempfile
                    r = requests.get(source, timeout=60)
                    r.raise_for_status()
                    fd, tmp_path = tempfile.mkstemp(suffix=".db")
                    with os.fdopen(fd, "wb") as f:
                        f.write(r.content)
                    db_path = tmp_path
                except Exception as e:
                    print(f"[warn] Could not download DB for preview: {e}")
                    raise
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(db_path)) as conn:
                if table is None:
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
                        conn
                    )
                    if tables.empty:
                        raise RuntimeError("No tables found in SQLite DB.")
                    table_name = tables.iloc[0, 0]
                else:
                    table_name = table
                return pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 100000;', conn)
        return pd.read_csv(source, nrows=50000)

    # Load preview dataframe to infer types and choose features
    try:
        df_preview = load_preview(args.input, args.table)
    except Exception as e:
        print(f"[error] Failed to load preview from {args.input}: {e}")
        sys.exit(1)

    # Infer types
    cats, nums = infer_feature_types(df_preview, max_unique_for_categorical=40)
    cats = [c for c in cats if df_preview[c].nunique(dropna=True) <= 1000]

    # Remove targets from lists to avoid duplication in sections (optional)
    for t in targets:
        if t in cats:
            try:
                cats.remove(t)
            except ValueError:
                pass
        if t in nums:
            try:
                nums.remove(t)
            except ValueError:
                pass

    # Limit counts for cleanliness
    cats = cats[: args.max_cat]
    nums = nums[: args.max_num]

    # Build notebook
    build_notebook(
        input_source=args.input,
        output_path=args.output,
        targets=targets,
        table=args.table,
        features_categorical=cats,
        features_numeric=nums,
    )

if __name__ == "__main__":
    main()
