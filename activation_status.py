"""
activation_status.py
--------------------
Adds ACTIVATION_CHURN and ACTIVATION_CHURN_TYPE columns to the latest
prime snapshot, based on the ACTIVATED column (values: A, I, D).

Logic
-----
For each RIMNO we look at the ACTIVATED column across all monthly files:

  - A RIMNO is considered "was_activated" if it had at least 1 row with
    ACTIVATED = 'A' in ANY historical month.

  - We then check the LAST TWO months (the latest file and the month
    before it).  If ALL rows for that RIMNO in BOTH of those months have
    ACTIVATED = 'I' or 'D', the customer has become deactivated.

ACTIVATION_CHURN_TYPE values
----------------------------
  deactivated_recent – RIMNO had 'A' in at least one historical month AND
                       all entries in the last two months are 'I' or 'D'.
                       High-confidence deactivation.
  always_inactive    – RIMNO NEVER had an 'A' in any month (including
                       the latest).  Was never truly activated.
  disappeared        – RIMNO appeared in historical files but is completely
                       absent from the latest file.
  new_customer       – RIMNO appears only in the latest file; no history.
  still_active       – RIMNO has at least one 'A' in the latest file
                       (still active, not churned).
  recently_active    – RIMNO had 'A' in the last month or month-before,
                       but all entries in the latest file are 'I'/'D'.
                       Very recent deactivation (single-month drop).

ACTIVATION_CHURN column
-----------------------
  1 for deactivated_recent, always_inactive, disappeared.
  0 for still_active, new_customer, recently_active.
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
OUTPUT_PATH    = "output/latest_prime_with_activation_churn.csv"

# Column names
CUSTOMER_ID_COL      = "RIMNO"         # after rename from RIM_NO
ACTIVATED_COL        = "ACTIVATED"
CHURN_COL            = "ACTIVATION_CHURN"
CHURN_TYPE_COL       = "ACTIVATION_CHURN_TYPE"

# Values in the ACTIVATED column
ACTIVE_VALUE         = "A"
INACTIVE_VALUES      = {"I", "D"}      # considered inactive / deactivated

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

month_map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}


def get_month_year(file_path):
    """Extracts proper datetime from filename like: cleaned_JUL_2025.csv"""
    name = os.path.basename(file_path).upper()
    match = re.search(
        r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_(\d{4})", name
    )
    if match:
        month = month_map[match.group(1)]
        year = int(match.group(2))
        return datetime(year, month, 1)
    return datetime(1900, 1, 1)


def get_month_label(file_path):
    """Return a human-readable label like '2025-JUL'."""
    name = os.path.basename(file_path).upper()
    match = re.search(
        r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_(\d{4})", name
    )
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
    # Normalise ACTIVATED to uppercase, strip whitespace
    if ACTIVATED_COL in df.columns:
        df[ACTIVATED_COL] = df[ACTIVATED_COL].astype(str).str.strip().str.upper()
    # Normalise RIMNO
    if CUSTOMER_ID_COL in df.columns:
        df[CUSTOMER_ID_COL] = (
            df[CUSTOMER_ID_COL].astype(str).str.strip().str.replace(",", "")
        )
    return df


def _is_inactive(val: str) -> bool:
    """Return True if the ACTIVATED value is inactive (I or D)."""
    return val.upper().strip() in INACTIVE_VALUES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def label_activation_churn():
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # 1. Discover and sort all prime CSVs chronologically
    all_files = glob.glob(os.path.join(PRIME_DATA_DIR, "*.csv"))
    if len(all_files) < 2:
        raise ValueError(
            f"Need at least 2 prime CSV files in '{PRIME_DATA_DIR}'; "
            f"found {len(all_files)}."
        )

    all_files.sort(key=get_month_year)

    latest_file    = all_files[-1]
    history_files  = all_files[:-1]

    # The "last two months" = latest file + the month right before it.
    # The month-before-latest is the last history file.
    last_month_file   = history_files[-1] if len(history_files) >= 1 else None
    month_before_file = history_files[-2] if len(history_files) >= 2 else None

    print("=" * 60)
    print("ACTIVATION STATUS ANALYSIS")
    print("=" * 60)
    print(f"Latest file         : {os.path.basename(latest_file)} ({get_month_label(latest_file)})")
    if last_month_file:
        print(f"Last month file     : {os.path.basename(last_month_file)} ({get_month_label(last_month_file)})")
    if month_before_file:
        print(f"Month-before file   : {os.path.basename(month_before_file)} ({get_month_label(month_before_file)})")
    print(f"Historical files    : {[f'{os.path.basename(f)} ({get_month_label(f)})' for f in history_files]}")
    print()

    # 2. Scan all historical files
    #    Track:
    #      - historical_rimnos: every RIMNO seen in history
    #      - ever_activated:    RIMNOs that had ACTIVATED='A' at least once
    #      - active_in_last_month: RIMNOs with 'A' in the last month file
    #      - active_in_month_before: RIMNOs with 'A' in the month-before file
    historical_rimnos: set[str]      = set()
    ever_activated: set[str]         = set()
    active_in_last_month: set[str]   = set()
    active_in_month_before: set[str] = set()

    for f in history_files:
        df = _load_csv(f)
        if CUSTOMER_ID_COL not in df.columns or ACTIVATED_COL not in df.columns:
            continue

        rimnos_in_file = set(df[CUSTOMER_ID_COL].dropna().unique())
        historical_rimnos.update(rimnos_in_file)

        # RIMNOs with at least one 'A' in this file
        a_mask = df[ACTIVATED_COL] == ACTIVE_VALUE
        activated_in_file = set(
            df.loc[a_mask, CUSTOMER_ID_COL].dropna().unique()
        )
        ever_activated.update(activated_in_file)

        # Track the two most recent months specifically
        if f == last_month_file:
            active_in_last_month = activated_in_file.copy()
        if f == month_before_file:
            active_in_month_before = activated_in_file.copy()

    # RIMNOs that were NEVER activated in any historical file
    never_activated_historically = historical_rimnos - ever_activated

    print(f"Unique RIMNOs in historical files  : {len(historical_rimnos):,}")
    print(f"  Ever activated (had 'A')         : {len(ever_activated):,}")
    print(f"  Active in last month             : {len(active_in_last_month):,}")
    print(f"  Active in month-before           : {len(active_in_month_before):,}")
    print(f"  Never activated historically     : {len(never_activated_historically):,}")
    print()

    # 3. Load latest file
    latest = _load_csv(latest_file)
    print(f"Rows in latest file                : {len(latest):,}")

    # Initialise columns
    latest[CHURN_COL]      = 0
    latest[CHURN_TYPE_COL] = "still_active"

    rimnos_in_latest = set(latest[CUSTOMER_ID_COL].dropna().unique())
    rimnos_in_both   = historical_rimnos & rimnos_in_latest

    # 4a. Check which RIMNOs have at least one 'A' in the latest file
    if ACTIVATED_COL in latest.columns:
        latest_a_mask = latest[ACTIVATED_COL] == ACTIVE_VALUE
        rimnos_active_in_latest = set(
            latest.loc[latest_a_mask, CUSTOMER_ID_COL].dropna().unique()
        )

        # RIMNOs where ALL entries in latest are I or D (fully inactive)
        inactive_mask_per_rimno = (
            latest[latest[CUSTOMER_ID_COL].isin(rimnos_in_both)]
            .groupby(CUSTOMER_ID_COL)[ACTIVATED_COL]
            .apply(lambda vals: vals.map(_is_inactive).all())
        )
        fully_inactive_in_latest = set(
            inactive_mask_per_rimno[inactive_mask_per_rimno].index.tolist()
        )
    else:
        print(f"WARNING: '{ACTIVATED_COL}' column not found in latest file — "
              "all customers set to ACTIVATION_CHURN=0.")
        rimnos_active_in_latest = set()
        fully_inactive_in_latest = set()

    # Also check: is the RIMNO fully inactive in the last_month_file?
    # We need to load the last month again to check per-RIMNO "all inactive"
    fully_inactive_in_last_month: set[str] = set()
    if last_month_file:
        lm_df = _load_csv(last_month_file)
        if CUSTOMER_ID_COL in lm_df.columns and ACTIVATED_COL in lm_df.columns:
            lm_inactive = (
                lm_df.groupby(CUSTOMER_ID_COL)[ACTIVATED_COL]
                .apply(lambda vals: vals.map(_is_inactive).all())
            )
            fully_inactive_in_last_month = set(
                lm_inactive[lm_inactive].index.tolist()
            )

    # 4b. Classify RIMNOs that exist in both history and latest
    #     Condition for deactivated_recent:
    #       - Was activated (had 'A') in at least one historical month
    #       - ALL entries in the latest file are I/D
    #       - ALL entries in the last month file are I/D
    #     (i.e. inactive in the last TWO months)

    # First: always_inactive — never had 'A' in any month including latest
    always_inactive = (
        (fully_inactive_in_latest & never_activated_historically)
        - rimnos_active_in_latest
    )

    # RIMNOs that were once activated but are now fully inactive in latest
    was_activated_now_inactive = fully_inactive_in_latest & ever_activated

    # Split by whether they were also inactive in the last month
    # (confirming the deactivation spans at least 2 months)
    deactivated_recent = (
        was_activated_now_inactive & fully_inactive_in_last_month
    )
    recently_active = (
        was_activated_now_inactive - fully_inactive_in_last_month
    )

    # 4c. Apply labels
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(always_inactive), CHURN_COL
    ] = 1
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(always_inactive), CHURN_TYPE_COL
    ] = "always_inactive"

    latest.loc[
        latest[CUSTOMER_ID_COL].isin(deactivated_recent), CHURN_COL
    ] = 1
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(deactivated_recent), CHURN_TYPE_COL
    ] = "deactivated_recent"

    # recently_active: was active in last month but inactive in latest
    # → single-month drop, might be temporary — CHURN=0 to be cautious
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(recently_active), CHURN_COL
    ] = 0
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(recently_active), CHURN_TYPE_COL
    ] = "recently_active"

    # 4d. RIMNOs only in latest (no history) → new_customer
    rimnos_only_in_latest = rimnos_in_latest - historical_rimnos
    latest.loc[
        latest[CUSTOMER_ID_COL].isin(rimnos_only_in_latest), CHURN_TYPE_COL
    ] = "new_customer"

    print()
    print(f"  Historical RIMNOs found in latest        : {len(rimnos_in_both):,}")
    print(f"    → still_active (has 'A' in latest)     : {len(rimnos_active_in_latest & rimnos_in_both):,}")
    print(f"    → deactivated_recent (I/D last 2 mo)   : {len(deactivated_recent):,}")
    print(f"    → recently_active (A last mo, I/D now)  : {len(recently_active):,}")
    print(f"    → always_inactive (never had 'A')       : {len(always_inactive):,}")
    print(f"  New customers (no history)                : {len(rimnos_only_in_latest):,}")

    # 4e. RIMNOs that are in historical files but MISSING from latest
    rimnos_missing = historical_rimnos - rimnos_in_latest
    print(f"  Disappeared (missing from latest)         : {len(rimnos_missing):,}")

    if rimnos_missing:
        stub_rows = pd.DataFrame(
            {CUSTOMER_ID_COL: list(rimnos_missing)}
        )
        stub_rows[CHURN_COL]      = 1
        stub_rows[CHURN_TYPE_COL] = "disappeared"

        for col in latest.columns:
            if col not in stub_rows.columns:
                stub_rows[col] = np.nan

        stub_rows = stub_rows[latest.columns]
        latest = pd.concat([latest, stub_rows], ignore_index=True)
        print(f"  Stub rows appended for missing RIMNOs    : {len(stub_rows):,}")

    # 5. Summary
    print()
    print("=" * 60)
    print("ACTIVATION CHURN SUMMARY")
    print("=" * 60)

    total_rows = len(latest)
    type_counts = latest[CHURN_TYPE_COL].value_counts()
    for ctype in ["deactivated_recent", "always_inactive", "disappeared",
                  "recently_active", "still_active", "new_customer"]:
        cnt = type_counts.get(ctype, 0)
        pct = cnt / total_rows * 100 if total_rows else 0
        churn_flag = (
            "CHURN=1" if ctype in ("deactivated_recent", "always_inactive", "disappeared")
            else "CHURN=0"
        )
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
    result = label_activation_churn()
