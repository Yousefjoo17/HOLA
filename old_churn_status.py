"""
churn_labeller.py
-----------------
Adds CHURN and CHURN_TYPE columns to the latest prime snapshot.

CHURN_TYPE values
-----------------
  always_churned  – RIMNO had a churn status in EVERY historical month AND
                    is still churned in the latest file.  These customers
                    were never really active.
  became_churned_recent – RIMNO was active (non-churn status) in at least
                     ONE of the last two months before the latest file AND
                     is now fully churned in the latest file.  High-confidence
                     recent churn.
  became_churned_old    – RIMNO was active at some point in history but was
                     NOT active in either of the last two months.  The churn
                     happened further back; flag for review.
  disappeared     – RIMNO appeared in historical files but is completely
                    absent from the latest file (stub row appended).
  new_customer    – RIMNO appears only in the latest file; no history.
  not_churned     – RIMNO exists in history and latest, and at least one
                    row in the latest file has a non-churn status.

CHURN column
------------
  1 for always_churned, became_churned_recent, became_churned_old,
    and disappeared.
  0 for not_churned and new_customer.

Adjust CHURN_STATUSES and PRIME_DATA_DIR as needed.
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

PRIME_DATA_DIR = "prime_data"          # folder containing all prime CSVs
OUTPUT_PATH    = "output/latest_prime_with_churn.csv"

# Statuses that, when held by ALL rows of a RIMNO in the latest snapshot,
# indicate the customer has churned.
CHURN_STATUSES = {
    "CLSB",   # closed by bank
    "CLSC",   # closed by customer
    "CLSD",   # closed
    "WROF",   # write-off
    "EXPD",   # expired (account)
    "EXPC",   # expired card
    "LOST",   # lost card (card no longer usable)
    "STLC",   # stolen card
    "FRAD",   # fraud pick-up
    "PICK",   # pick-up card
}

# Column names
CUSTOMER_ID_COL = "RIMNO"          # after rename from RIM_NO
STATUS_COL      = "STATUS"
CHURN_COL       = "CHURN"
CHURN_TYPE_COL  = "CHURN_TYPE"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _normalise_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """Rename RIM_NO → RIMNO if needed."""
    if "RIM_NO" in df.columns and CUSTOMER_ID_COL not in df.columns:
        df = df.rename(columns={"RIM_NO": CUSTOMER_ID_COL})
    return df


def _load_csv(path: str) -> pd.DataFrame:
    """Read a single prime CSV with robust encoding."""
    df = pd.read_csv(path, encoding="latin", dtype=str, low_memory=False)
    df = _normalise_id_col(df)
    # Normalise STATUS to uppercase, strip whitespace
    if STATUS_COL in df.columns:
        df[STATUS_COL] = df[STATUS_COL].astype(str).str.strip().str.upper()
    # Normalise RIMNO
    if CUSTOMER_ID_COL in df.columns:
        df[CUSTOMER_ID_COL] = (
            df[CUSTOMER_ID_COL].astype(str).str.strip().str.replace(",", "")
        )
    return df


def _is_churned_status(status: str) -> bool:
    return status.upper().strip() in CHURN_STATUSES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def label_churn():
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # 1. Discover and sort all prime CSVs
    all_files = glob.glob(os.path.join(PRIME_DATA_DIR, "*.csv"))
    if len(all_files) < 2:
        raise ValueError(
            f"Need at least 2 prime CSV files in '{PRIME_DATA_DIR}'; "
            f"found {len(all_files)}."
        )

    all_files.sort(key=get_month_year)

    latest_file    = all_files[-1]
    history_files  = all_files[:-1]

    print(f"Latest prime file   : {os.path.basename(latest_file)} ({get_month_label(latest_file)})")
    print(f"Historical files    : {[f'{os.path.basename(f)} ({get_month_label(f)})' for f in history_files]}")
    print()

    # 2. Load historical files and track per-RIMNO status across months
    #    For each RIMNO we record:
    #      - was it ever active (non-churn status) in any historical month?
    #      - was it active in the last month or the month before that?
    #      - which months did it appear in?
    historical_rimnos: set[str] = set()
    ever_active_rimnos: set[str] = set()   # had a non-churn status at least once

    # Identify the last two history files (the month right before latest,
    # and the month before that) for the extra-cautious recency check.
    last_month_file       = history_files[-1] if len(history_files) >= 1 else None
    month_before_file     = history_files[-2] if len(history_files) >= 2 else None
    recent_active_rimnos: set[str] = set()  # active in last month or month-before

    if last_month_file:
        print(f"Last month file     : {os.path.basename(last_month_file)} ({get_month_label(last_month_file)})")
    if month_before_file:
        print(f"Month-before file   : {os.path.basename(month_before_file)} ({get_month_label(month_before_file)})")
    print()

    for f in history_files:
        df = _load_csv(f)
        if CUSTOMER_ID_COL not in df.columns:
            continue
        rimnos_in_file = set(df[CUSTOMER_ID_COL].dropna().unique())
        historical_rimnos.update(rimnos_in_file)

        # Check which RIMNOs in this file have at least one non-churn row
        if STATUS_COL in df.columns:
            active_mask = ~df[STATUS_COL].map(_is_churned_status)
            active_in_file = set(
                df.loc[active_mask, CUSTOMER_ID_COL].dropna().unique()
            )
            ever_active_rimnos.update(active_in_file)

            # Extra-cautious: track activity in the two most recent months
            if f in (last_month_file, month_before_file):
                recent_active_rimnos.update(active_in_file)

    # RIMNOs that were ALWAYS churned in every historical file they appeared in
    always_churned_historically = historical_rimnos - ever_active_rimnos

    print(f"Unique RIMNOs in historical files : {len(historical_rimnos):,}")
    print(f"  Ever active in history          : {len(ever_active_rimnos):,}")
    print(f"  Active in last 2 months         : {len(recent_active_rimnos):,}")
    print(f"  Always churned in history       : {len(always_churned_historically):,}")

    # 3. Load latest prime
    latest = _load_csv(latest_file)
    print(f"Rows in latest prime              : {len(latest):,}")

    # Initialise columns
    latest[CHURN_COL]      = 0
    latest[CHURN_TYPE_COL] = "not_churned"

    # 4. For each historical RIMNO, decide CHURN and CHURN_TYPE
    rimnos_in_latest = set(latest[CUSTOMER_ID_COL].dropna().unique())
    rimnos_in_both   = historical_rimnos & rimnos_in_latest

    # 4a. Determine which RIMNOs are fully churned in the latest file
    if STATUS_COL in latest.columns:
        churn_mask_per_rimno = (
            latest[latest[CUSTOMER_ID_COL].isin(rimnos_in_both)]
            .groupby(CUSTOMER_ID_COL)[STATUS_COL]
            .apply(lambda statuses: statuses.map(_is_churned_status).all())
        )
        churned_in_latest = set(
            churn_mask_per_rimno[churn_mask_per_rimno].index.tolist()
        )
    else:
        print(f"WARNING: '{STATUS_COL}' column not found in latest prime — "
              "all existing customers set to CHURN=0.")
        churned_in_latest = set()

    # 4b. Split churned RIMNOs into "always_churned" vs "became_churned"
    #      For became_churned, further split by recency:
    #        - became_churned_recent: was active in the last month or month before
    #        - became_churned_old:    was active historically but NOT recently
    always_churned        = churned_in_latest & always_churned_historically
    became_churned_all    = churned_in_latest - always_churned_historically
    became_churned_recent = became_churned_all & recent_active_rimnos
    became_churned_old    = became_churned_all - recent_active_rimnos

    # Apply labels
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(always_churned), CHURN_COL
    ] = 1
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(always_churned), CHURN_TYPE_COL
    ] = "always_churned"

    latest.loc[
        latest[CUSTOMER_ID_COL].isin(became_churned_recent), CHURN_COL
    ] = 1
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(became_churned_recent), CHURN_TYPE_COL
    ] = "became_churned_recent"

    latest.loc[
        latest[CUSTOMER_ID_COL].isin(became_churned_old), CHURN_COL
    ] = 1
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(became_churned_old), CHURN_TYPE_COL
    ] = "became_churned_old"

    # 4c. RIMNOs only in latest (no history) → new_customer
    rimnos_only_in_latest = rimnos_in_latest - historical_rimnos
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(rimnos_only_in_latest), CHURN_TYPE_COL
    ] = "new_customer"

    print()
    print(f"  Historical RIMNOs found in latest    : {len(rimnos_in_both):,}")
    print(f"    → always_churned (churned every month)      : {len(always_churned):,}")
    print(f"    → became_churned_recent (active last 2 mo)  : {len(became_churned_recent):,}")
    print(f"    → became_churned_old (active only earlier)   : {len(became_churned_old):,}")
    print(f"    → not_churned (still active)                : {len(rimnos_in_both) - len(churned_in_latest):,}")
    print(f"  New customers (no history)                    : {len(rimnos_only_in_latest):,}")

    # 4d. RIMNOs that are in historical files but MISSING from latest_prime
    rimnos_missing = historical_rimnos - rimnos_in_latest
    print(f"  Disappeared (missing from latest)     : {len(rimnos_missing):,}")

    if rimnos_missing:
        # Build stub rows: one row per missing RIMNO, all columns NaN
        # except RIMNO, CHURN, and CHURN_TYPE.
        stub_rows = pd.DataFrame(
            {CUSTOMER_ID_COL: list(rimnos_missing)}
        )
        stub_rows[CHURN_COL]      = 1
        stub_rows[CHURN_TYPE_COL] = "disappeared"

        # Add any columns present in latest but absent in stub (fill NaN)
        for col in latest.columns:
            if col not in stub_rows.columns:
                stub_rows[col] = np.nan

        # Re-order columns to match latest
        stub_rows = stub_rows[latest.columns]

        latest = pd.concat([latest, stub_rows], ignore_index=True)
        print(f"  Stub rows appended for missing RIMNOs : {len(stub_rows):,}")

    # 5. Summary
    print()
    print("=" * 60)
    print("CHURN SUMMARY")
    print("=" * 60)

    total_rows = len(latest)
    type_counts = latest[CHURN_TYPE_COL].value_counts()
    for ctype in ["always_churned", "became_churned_recent", "became_churned_old",
                  "disappeared", "not_churned", "new_customer"]:
        cnt = type_counts.get(ctype, 0)
        pct = cnt / total_rows * 100 if total_rows else 0
        churn_flag = "CHURN=1" if ctype in ("always_churned", "became_churned_recent", "became_churned_old", "disappeared") else "CHURN=0"
        print(f"  {ctype:<25s}: {cnt:>8,}  ({pct:5.1f}%)  [{churn_flag}]")

    total_churn = int(latest[CHURN_COL].sum())
    print(f"  {'':20s}  {'':>8s}")
    print(f"  {'TOTAL':20s}: {total_rows:>8,}")
    print(f"  {'CHURN = 1':20s}: {total_churn:>8,}  ({total_churn / total_rows * 100:.1f}%)")
    print(f"  {'CHURN = 0':20s}: {total_rows - total_churn:>8,}  ({(total_rows - total_churn) / total_rows * 100:.1f}%)")
    print()

    # 6. Write output
    latest.to_csv(OUTPUT_PATH, index=False)
    print(f"Output saved to: {OUTPUT_PATH}")

    return latest


if __name__ == "__main__":
    result = label_churn()