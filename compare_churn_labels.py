"""
compare_churn_labels.py
-----------------------
Compares the churn labels produced by churn_status.py (STATUS-based)
and activation_status.py (ACTIVATED-based) to highlight differences.

Reads:
  - output/latest_prime_with_churn.csv            (CHURN, CHURN_TYPE)
  - output/latest_prime_with_activation_churn.csv  (ACTIVATION_CHURN, ACTIVATION_CHURN_TYPE)

Produces:
  - output/churn_comparison.csv   (merged view with disagreement flag)
  - Console summary of agreements / disagreements
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATUS_CHURN_PATH     = "output/latest_prime_with_churn.csv"
ACTIVATION_CHURN_PATH = "output/latest_prime_with_activation_churn.csv"
OUTPUT_PATH           = "output/churn_comparison.csv"

CUSTOMER_ID_COL = "RIMNO"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare():
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # 1. Load both outputs
    if not os.path.exists(STATUS_CHURN_PATH):
        raise FileNotFoundError(
            f"STATUS-based churn file not found: {STATUS_CHURN_PATH}\n"
            "Run churn_status.py first."
        )
    if not os.path.exists(ACTIVATION_CHURN_PATH):
        raise FileNotFoundError(
            f"ACTIVATION-based churn file not found: {ACTIVATION_CHURN_PATH}\n"
            "Run activation_status.py first."
        )

    status_df = pd.read_csv(STATUS_CHURN_PATH, dtype=str, low_memory=False)
    activ_df  = pd.read_csv(ACTIVATION_CHURN_PATH, dtype=str, low_memory=False)

    # Normalise RIMNO
    for df in [status_df, activ_df]:
        df[CUSTOMER_ID_COL] = (
            df[CUSTOMER_ID_COL].astype(str).str.strip().str.replace(",", "")
        )

    # Keep only the relevant columns from each
    status_cols = [CUSTOMER_ID_COL, "CHURN", "CHURN_TYPE"]
    activ_cols  = [CUSTOMER_ID_COL, "ACTIVATION_CHURN", "ACTIVATION_CHURN_TYPE"]

    # Deduplicate to one row per RIMNO (take the first / worst label)
    # For STATUS churn: if ANY row for the RIMNO says CHURN=1, treat as churned
    status_summary = (
        status_df[status_cols]
        .drop_duplicates(subset=CUSTOMER_ID_COL)
    )
    activ_summary = (
        activ_df[activ_cols]
        .drop_duplicates(subset=CUSTOMER_ID_COL)
    )

    # 2. Merge on RIMNO
    merged = pd.merge(
        status_summary,
        activ_summary,
        on=CUSTOMER_ID_COL,
        how="outer",
        indicator=True,
    )

    # Convert churn flags to int for comparison
    merged["CHURN"] = pd.to_numeric(merged["CHURN"], errors="coerce").fillna(-1).astype(int)
    merged["ACTIVATION_CHURN"] = pd.to_numeric(merged["ACTIVATION_CHURN"], errors="coerce").fillna(-1).astype(int)

    # 3. Flag disagreements
    merged["AGREE"] = merged["CHURN"] == merged["ACTIVATION_CHURN"]
    merged["DISAGREE_TYPE"] = ""

    # STATUS says churn, ACTIVATION says no churn
    mask_status_only = (merged["CHURN"] == 1) & (merged["ACTIVATION_CHURN"] == 0)
    merged.loc[mask_status_only, "DISAGREE_TYPE"] = "STATUS_churn_only"

    # ACTIVATION says churn, STATUS says no churn
    mask_activ_only = (merged["CHURN"] == 0) & (merged["ACTIVATION_CHURN"] == 1)
    merged.loc[mask_activ_only, "DISAGREE_TYPE"] = "ACTIVATION_churn_only"

    # Both agree churn
    mask_both_churn = (merged["CHURN"] == 1) & (merged["ACTIVATION_CHURN"] == 1)
    merged.loc[mask_both_churn, "DISAGREE_TYPE"] = "both_churn"

    # Both agree no churn
    mask_both_ok = (merged["CHURN"] == 0) & (merged["ACTIVATION_CHURN"] == 0)
    merged.loc[mask_both_ok, "DISAGREE_TYPE"] = "both_not_churn"

    # Only in one file
    merged.loc[merged["_merge"] == "left_only", "DISAGREE_TYPE"] = "only_in_status_file"
    merged.loc[merged["_merge"] == "right_only", "DISAGREE_TYPE"] = "only_in_activation_file"

    # 4. Print summary
    total = len(merged)
    print("=" * 65)
    print("CHURN LABEL COMPARISON: STATUS vs ACTIVATED")
    print("=" * 65)
    print(f"Total unique RIMNOs across both files : {total:,}")
    print(f"  In both files                       : {(merged['_merge'] == 'both').sum():,}")
    print(f"  Only in STATUS file                 : {(merged['_merge'] == 'left_only').sum():,}")
    print(f"  Only in ACTIVATION file             : {(merged['_merge'] == 'right_only').sum():,}")
    print()

    # Agreement stats (only for RIMNOs in both files)
    both_mask = merged["_merge"] == "both"
    both_df   = merged[both_mask]
    agree     = both_df["AGREE"].sum()
    disagree  = (~both_df["AGREE"]).sum()

    print(f"Among RIMNOs in BOTH files ({len(both_df):,}):")
    print(f"  Agree (same label)                  : {agree:,}  ({agree / len(both_df) * 100:.1f}%)")
    print(f"  Disagree                            : {disagree:,}  ({disagree / len(both_df) * 100:.1f}%)")
    print()

    # Breakdown of disagreements
    print("Detailed breakdown:")
    type_counts = merged["DISAGREE_TYPE"].value_counts()
    for dtype in ["both_churn", "both_not_churn", "STATUS_churn_only",
                  "ACTIVATION_churn_only", "only_in_status_file",
                  "only_in_activation_file"]:
        cnt = type_counts.get(dtype, 0)
        pct = cnt / total * 100 if total else 0
        print(f"  {dtype:<30s}: {cnt:>8,}  ({pct:5.1f}%)")

    print()

    # Cross-tab of churn types
    print("CHURN_TYPE vs ACTIVATION_CHURN_TYPE cross-tab:")
    print("-" * 65)
    ct = pd.crosstab(
        merged["CHURN_TYPE"].fillna("(missing)"),
        merged["ACTIVATION_CHURN_TYPE"].fillna("(missing)"),
        margins=True,
    )
    print(ct.to_string())
    print()

    # 5. Show sample disagreements
    disagreements = merged[~merged["AGREE"] & both_mask]
    if len(disagreements) > 0:
        print(f"Sample disagreements (up to 20):")
        print("-" * 65)
        sample = disagreements.head(20)[
            [CUSTOMER_ID_COL, "CHURN", "CHURN_TYPE",
             "ACTIVATION_CHURN", "ACTIVATION_CHURN_TYPE"]
        ]
        print(sample.to_string(index=False))
    else:
        print("No disagreements found — both methods agree on every RIMNO!")

    print()

    # 6. Save full comparison
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Full comparison saved to: {OUTPUT_PATH}")

    return merged


if __name__ == "__main__":
    result = compare()
