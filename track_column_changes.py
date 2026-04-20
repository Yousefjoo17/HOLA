"""
Track Column Changes Across Months
===================================
For each unique (RIMNO, PRODUCT_NAME) combination, this script checks
which other columns change their value from one month to the next.

Expects CSV files in the 'prime/' subdirectory, named so that they sort
chronologically (e.g. 2025JUL.csv, 2025AUG.csv, ..., 2026FEB.csv).

Output:
  1. Console summary of which columns change and how often.
  2. A CSV file 'column_changes_report.csv' with full details.
  3. A CSV file 'column_changes_summary.csv' with aggregate stats.
"""

import glob
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np

# ── 1. Discover and order files chronologically ──────────────────────────
month_map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def get_month_year(file_path):
    """
    Extracts proper datetime from filename like:
    cleaned_JUL_2025.csv
    """
    name = os.path.basename(file_path).upper()
    match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_(\d{4})", name)
    
    if match:
        month = month_map[match.group(1)]
        year = int(match.group(2))
        return datetime(year, month, 1)
    
    # fallback very early date
    return datetime(1900, 1, 1)

def get_month_label(file_path):
    """Return a human-readable label like '2025-JUL'."""
    name = os.path.basename(file_path).upper()
    match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_(\d{4})", name)
    if match:
        return f"{match.group(2)}-{match.group(1)}"
    return os.path.splitext(os.path.basename(file_path))[0]

prime_files = glob.glob("prime/*.csv")
if not prime_files:
    print("ERROR: No CSV files found in 'prime/' folder. "
          "Make sure you run this script from the directory that contains 'prime/'.")
    raise SystemExit(1)

prime_files.sort(key=get_month_year)
print(f"Found {len(prime_files)} files, ordered as:")
for f in prime_files:
    print(f"  {get_month_label(f)}  ←  {f}")

# ── 2. Load a single file and normalise columns ─────────────────────────
def load_month(filepath):
    """Load a CSV, normalise known column variants, return DataFrame."""
    df = pd.read_csv(filepath, encoding="latin", dtype=str)
    if "RIM_NO" in df.columns:
        df = df.rename(columns={"RIM_NO": "RIMNO"})
    if "NAME" in df.columns:
        df = df.rename(columns={"NAME": "PRODUCT_NAME"})
    return df

KEY_COLS = ["RIMNO", "PRODUCT_NAME"]

# ── 3. Compare each consecutive pair of months ──────────────────────────
all_change_records = []
total_pairs_seen = set()

for i in range(len(prime_files) - 1):
    file_a, file_b = prime_files[i], prime_files[i + 1]
    label_a, label_b = get_month_label(file_a), get_month_label(file_b)

    print(f"\n{'='*70}")
    print(f"Comparing  {label_a}  -->  {label_b}")
    print(f"{'='*70}")

    df_a = load_month(file_a)
    df_b = load_month(file_b)

    # Columns to compare = intersection of both files minus the keys
    common_cols = [c for c in df_a.columns if c in df_b.columns and c not in KEY_COLS]
    print(f"  Common columns to compare: {len(common_cols)}")

    # Inner merge on the key so we only compare rows that exist in both months
    merged = df_a[KEY_COLS + common_cols].merge(
        df_b[KEY_COLS + common_cols],
        on=KEY_COLS,
        suffixes=("_OLD", "_NEW"),
        how="inner",
    )
    print(f"  Matched (RIMNO, PRODUCT_NAME) pairs: {len(merged):,}")

    change_records = []
    for col in common_cols:
        old_col = f"{col}_OLD"
        new_col = f"{col}_NEW"

        old_vals = merged[old_col]
        new_vals = merged[new_col]

        # A value changed if they differ AND it's not NaN==NaN
        both_nan = old_vals.isna() & new_vals.isna()
        changed = (old_vals != new_vals) & ~both_nan

        if changed.any():
            diff_rows = merged.loc[changed, KEY_COLS + [old_col, new_col]].copy()
            diff_rows = diff_rows.rename(columns={old_col: "OLD_VALUE", new_col: "NEW_VALUE"})
            diff_rows["COLUMN"] = col
            diff_rows["FROM_MONTH"] = label_a
            diff_rows["TO_MONTH"] = label_b
            change_records.append(diff_rows)

    if change_records:
        transition_df = pd.concat(change_records, ignore_index=True)
        all_change_records.append(transition_df)

        # Per-column summary for this transition
        col_counts = transition_df["COLUMN"].value_counts()
        print(f"\n  Columns that changed ({label_a} -> {label_b}):")
        for col_name, count in col_counts.items():
            pct = count / len(merged) * 100
            print(f"    {col_name:30s}  {count:>7,} rows changed  ({pct:.1f}%)")
    else:
        print(f"  No changes detected between {label_a} and {label_b}.")

    total_pairs_seen.update(merged[KEY_COLS].apply(tuple, axis=1).tolist())

# ── 4. Aggregate report across all transitions ──────────────────────────
if all_change_records:
    changes_df = pd.concat(all_change_records, ignore_index=True)
    total_tracked = len(total_pairs_seen)

    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique (RIMNO, PRODUCT_NAME) pairs tracked: {total_tracked:,}")
    print(f"Total individual field changes detected:          {len(changes_df):,}")

    # Which columns change the most across all transitions?
    col_summary = (
        changes_df
        .groupby("COLUMN")
        .agg(
            total_changes=("COLUMN", "size"),
            unique_pairs_affected=("RIMNO", "nunique"),
        )
        .sort_values("total_changes", ascending=False)
    )
    col_summary["pct_pairs_affected"] = (
        (col_summary["unique_pairs_affected"] / total_tracked * 100).round(2)
    )

    print("\n-- Column-Level Summary (sorted by frequency) --")
    print(col_summary.to_string())

    # Export
    detail_path = "column_changes_report.csv"
    summary_path = "column_changes_summary.csv"

    changes_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    col_summary.to_csv(summary_path, encoding="utf-8-sig")

    print(f"\nDetailed change log  ->  {detail_path}")
    print(f"Column-level summary ->  {summary_path}")
else:
    print("\nNo changes detected across any month transition.")

print("\nDone.")
