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

# ── 2. Load each file, tag with its month ────────────────────────────────
frames = []
for filepath in prime_files:
    label = get_month_label(filepath)
    df = pd.read_csv(filepath, encoding="latin", dtype=str)  # read everything as string for safe comparison
    # Normalise the two known column-name variants
    if "RIM_NO" in df.columns:
        df = df.rename(columns={"RIM_NO": "RIMNO"})
    if "NAME" in df.columns:
        df = df.rename(columns={"NAME": "PRODUCT_NAME"})
    df["_MONTH"] = label
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows loaded: {len(combined):,}")
print(f"Unique (RIMNO, PRODUCT_NAME) pairs: "
      f"{combined.groupby(['RIMNO','PRODUCT_NAME']).ngroups:,}")

# ── 3. Identify the key and the "other" columns ─────────────────────────
KEY_COLS = ["RIMNO", "PRODUCT_NAME"]
SKIP_COLS = set(KEY_COLS) | {"_MONTH"}
compare_cols = [c for c in combined.columns if c not in SKIP_COLS]

print(f"\nColumns that will be compared month-over-month ({len(compare_cols)}):")
for c in compare_cols:
    print(f"  • {c}")

# ── 4. Detect changes ───────────────────────────────────────────────────
# For every (RIMNO, PRODUCT_NAME) group, sort by month and diff each
# column against the previous month.

change_records = []

grouped = combined.sort_values("_MONTH").groupby(KEY_COLS, sort=False)
total_groups = len(grouped)
print(f"\nAnalysing {total_groups:,} groups for changes …")

for idx, ((rimno, prod), grp) in enumerate(grouped, 1):
    if idx % 5000 == 0 or idx == total_groups:
        print(f"  progress: {idx:,} / {total_groups:,}")

    if len(grp) < 2:
        continue  # need at least 2 months to compare

    grp = grp.sort_values("_MONTH").reset_index(drop=True)
    months = grp["_MONTH"].tolist()

    for i in range(1, len(grp)):
        prev_month = months[i - 1]
        curr_month = months[i]
        for col in compare_cols:
            old_val = grp.at[i - 1, col]
            new_val = grp.at[i, col]
            # Treat NaN == NaN as "no change"
            if pd.isna(old_val) and pd.isna(new_val):
                continue
            if old_val != new_val:
                change_records.append({
                    "RIMNO": rimno,
                    "PRODUCT_NAME": prod,
                    "COLUMN": col,
                    "FROM_MONTH": prev_month,
                    "TO_MONTH": curr_month,
                    "OLD_VALUE": old_val,
                    "NEW_VALUE": new_val,
                })

changes_df = pd.DataFrame(change_records)

# ── 5. Report ────────────────────────────────────────────────────────────
if changes_df.empty:
    print("\nNo column changes detected across months for any (RIMNO, PRODUCT_NAME) pair.")
else:
    print(f"\n{'='*70}")
    print(f"TOTAL INDIVIDUAL CHANGES DETECTED: {len(changes_df):,}")
    print(f"{'='*70}")

    # 5a. Which columns change the most?
    col_summary = (
        changes_df
        .groupby("COLUMN")
        .agg(
            total_changes=("COLUMN", "size"),
            unique_pairs_affected=("RIMNO", "nunique"),
        )
        .sort_values("total_changes", ascending=False)
    )
    # What % of all tracked pairs are affected
    col_summary["pct_pairs_affected"] = (
        (col_summary["unique_pairs_affected"] / total_groups * 100).round(2)
    )

    print("\n── Column-Level Summary (sorted by frequency) ──")
    print(col_summary.to_string())

    # 5b. Month-transition breakdown
    transition_summary = (
        changes_df
        .groupby(["FROM_MONTH", "TO_MONTH", "COLUMN"])
        .size()
        .reset_index(name="change_count")
        .sort_values(["FROM_MONTH", "change_count"], ascending=[True, False])
    )
    print("\n── Changes by Month Transition ──")
    print(transition_summary.to_string(index=False))

    # 5c. Export detailed results
    detail_path = "column_changes_report.csv"
    summary_path = "column_changes_summary.csv"

    changes_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    col_summary.to_csv(summary_path, encoding="utf-8-sig")

    print(f"\nDetailed change log  ->  {detail_path}")
    print(f"Column-level summary ->  {summary_path}")

print("\nDone.")
