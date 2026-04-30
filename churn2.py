"""
Churn Labeling Script
=====================
Processes monthly CSV files from JUL2025 to FEB2026.
Logic: If a RIMNO exists in JUL2025 but NOT in FEB2026 → churn = 1, else churn = 0

Output columns: RIMNO, Card_account_Status, churn
"""

import pandas as pd
import os
import glob

# ─────────────────────────────────────────────
# CONFIGURATION — update these to match your setup
# ─────────────────────────────────────────────

# Folder containing all monthly CSV files
DATA_FOLDER = "."  # Change to your folder path if needed

# Column names in your CSV files (update if different)
RIMNO_COL = "RIMNO"
STATUS_COL = "Card_account_Status"

# File naming pattern — update to match your actual file names
# Examples: "data_JUL2025.csv", "JUL2025_data.csv", "2025-07.csv"
FILE_PATTERN = "*JUL2025*.csv"   # used only for auto-detection

# Explicit file paths (recommended — edit these to your actual filenames)
MONTHLY_FILES = {
    "JUL2025": "JUL2025.csv",
    "AUG2025": "AUG2025.csv",
    "SEP2025": "SEP2025.csv",
    "OCT2025": "OCT2025.csv",
    "NOV2025": "NOV2025.csv",
    "DEC2025": "DEC2025.csv",
    "JAN2026": "JAN2026.csv",
    "FEB2026": "FEB2026.csv",
}

OUTPUT_FILE = "churn_output.csv"

# ─────────────────────────────────────────────
# STEP 1: Load the two key months
# ─────────────────────────────────────────────

def load_month(filepath, label):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[{label}] File not found: {filepath}")
    df = pd.read_csv(filepath, usecols=[RIMNO_COL, STATUS_COL], dtype=str)
    df[RIMNO_COL] = df[RIMNO_COL].str.strip()
    df[STATUS_COL] = df[STATUS_COL].str.strip()
    df.dropna(subset=[RIMNO_COL], inplace=True)
    print(f"  [{label}] Loaded {len(df):,} rows | unique RIMNOs: {df[RIMNO_COL].nunique():,}")
    return df

print("=" * 55)
print("  Churn Labeling — JUL2025 → FEB2026")
print("=" * 55)

jul_path = os.path.join(DATA_FOLDER, MONTHLY_FILES["JUL2025"])
feb_path = os.path.join(DATA_FOLDER, MONTHLY_FILES["FEB2026"])

print("\nLoading monthly files...")
jul_df = load_month(jul_path, "JUL2025")
feb_df = load_month(feb_path, "FEB2026")

# ─────────────────────────────────────────────
# STEP 2: Build reference set from FEB2026
# ─────────────────────────────────────────────

feb_rimnos = set(feb_df[RIMNO_COL].unique())
print(f"\nFEB2026 unique RIMNOs (active set): {len(feb_rimnos):,}")

# ─────────────────────────────────────────────
# STEP 3: Label churn on JUL2025 base
# ─────────────────────────────────────────────

# Keep latest status per RIMNO in JUL2025 (in case of duplicates)
jul_deduped = jul_df.drop_duplicates(subset=[RIMNO_COL], keep="last").copy()

jul_deduped["churn"] = jul_deduped[RIMNO_COL].apply(
    lambda r: 1 if r not in feb_rimnos else 0
)

# ─────────────────────────────────────────────
# STEP 4: Summary
# ─────────────────────────────────────────────

total     = len(jul_deduped)
churned   = jul_deduped["churn"].sum()
retained  = total - churned
rate      = churned / total * 100 if total > 0 else 0

print("\n──────────────────────────────────────────────")
print("  CHURN SUMMARY")
print("──────────────────────────────────────────────")
print(f"  Total customers (JUL2025) : {total:>10,}")
print(f"  Churned (churn = 1)       : {churned:>10,}  ({rate:.2f}%)")
print(f"  Retained (churn = 0)      : {retained:>10,}  ({100-rate:.2f}%)")
print("──────────────────────────────────────────────")

# ─────────────────────────────────────────────
# STEP 5: Export output
# ─────────────────────────────────────────────

output = jul_deduped[[RIMNO_COL, STATUS_COL, "churn"]].reset_index(drop=True)
output.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Output saved → {OUTPUT_FILE}")
print(f"   Columns: {list(output.columns)}")
print(f"   Rows   : {len(output):,}")
print("\nPreview (first 10 rows):")
print(output.head(10).to_string(index=False))