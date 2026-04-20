"""
Credit Card Data Analysis — Visualization Report Generator
===========================================================
Mirrors the preprocessing pipeline in push.py and generates
presentation-ready plots for every major analysis stage.

Outputs are saved to  ./plots/  as high-resolution PNGs.
Run:  python generate_plots.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
})
PALETTE = sns.color_palette("coolwarm", 12)
OUT = "plots"
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print(f"  ✔ saved {name}")


# =====================================================================
# 1. LOAD DATA  (same logic as push.py)
# =====================================================================
print("▶ Loading prime data …")
prime_str = ["BRANCH_NAME","ACTIVATED","STATUS","STATUS_NAME","PRODUCT_NAME",
             "GENDER","ORGANIZATION","CUSTOMER_TYPE","Card account status "]
prime_int = ["BRANCH_ID","RIMNO"]
prime_flt = ["CREDIT_LIMIT","DELIQUENCY","JOINING_FEE","ANNUAL_FEE",
             "LEDGER_BALANCE","AVAILABLE_LIMIT","LAST_PAYMENT_AMOUNT",
             "OVERDUEAMOUNT","NO_OF_CYCLES","FIRST_REPLACED_CARD",
             "SECOND_REPLACED_CARD","THIRD_REPLACED_CARD","SETTLEMENT AMT"]
prime_dat = ["CREATION_DATE","LAST_STATEMENT_DATE","LAST_PAYMENT_DATE",
             "DOB","CLOSURE_DATE"]

prime_frames = []
for f in glob.glob("prime/*.csv"):
    tmp = pd.read_csv(
        f, encoding="latin",
        dtype={c: "string" for c in prime_str + prime_int + prime_flt},
        parse_dates=prime_dat
    ).rename(columns={"RIM_NO": "RIMNO", "NAME": "PRODUCT_NAME"})
    tmp["DOB"] = pd.to_datetime(tmp["DOB"], errors='coerce')
    prime_frames.append(tmp)
prime_df = pd.concat(prime_frames, ignore_index=True)

# cast floats
for c in prime_flt:
    if c in prime_df.columns:
        prime_df[c] = pd.to_numeric(
            prime_df[c].astype(str).str.replace(",", ""), errors="coerce"
        )
# cast ints
for c in prime_int:
    if c in prime_df.columns:
        prime_df[c] = pd.to_numeric(
            prime_df[c].astype(str).str.replace(",", ""), errors="coerce"
        ).astype("Int64")

# drop unused cols
for c in ["MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT", "TOTAL_HOLD"]:
    if c in prime_df.columns:
        prime_df.drop(columns=c, inplace=True)

print("▶ Loading transaction data …")
txn_str = ["DESCRIPTION","MERCHNAME","MERCH ID","SOURCES",
           "BANKBRANCH","TRXN COUNTRY","REVERSAL FLAG"]
txn_int = ["RIMNO","CCY","MCC","SETTLEMENT CCY"]
txn_flt = ["ORIG AMOUNT","EMBEDDED _FEE","BILLING AMT","SETTLEMENT AMT"]
txn_dat = ["TRXN DATE","POST DATE"]

txn_frames = []
for f in glob.glob("transaction/*.xlsx"):
    tmp = pd.read_excel(
        f,
        dtype={c: "string" for c in txn_str + txn_int + txn_flt},
        parse_dates=txn_dat
    )
    txn_frames.append(tmp)
transaction_df = pd.concat(txn_frames, ignore_index=True)

for c in txn_flt:
    if c in transaction_df.columns:
        transaction_df[c] = pd.to_numeric(
            transaction_df[c].astype(str).str.replace(",", ""), errors="coerce"
        )
for c in txn_int:
    if c in transaction_df.columns:
        transaction_df[c] = pd.to_numeric(
            transaction_df[c].astype(str).str.replace(",", ""), errors="coerce"
        ).astype("Int64")

# ── keep a raw copy for before/after outlier comparison ──
prime_raw = prime_df.copy()
txn_raw   = transaction_df.copy()

# =====================================================================
# 2. MISSING-VALUE ANALYSIS
# =====================================================================
print("\n▶ Plotting missing-value analysis …")

def plot_missing(df, title, fname):
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print(f"  (no missing values in {title})")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.4)))
    colors = ["#e74c3c" if v > 0.3 else "#f39c12" if v > 0.05 else "#2ecc71"
              for v in miss.values]
    bars = ax.barh(miss.index, miss.values * 100, color=colors, edgecolor="white")
    ax.set_xlabel("Missing %")
    ax.set_title(f"Missing Values — {title}")
    ax.invert_yaxis()
    for bar, pct in zip(bars, miss.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct*100:.1f}%", va="center", fontsize=9)
    save(fig, fname)

plot_missing(prime_df, "Prime (Customer) Data", "01_missing_prime.png")
plot_missing(transaction_df, "Transaction Data", "01_missing_transactions.png")

# =====================================================================
# 3. CATEGORICAL DISTRIBUTIONS
# =====================================================================
print("\n▶ Plotting categorical distributions …")

cat_cols_prime = ["GENDER", "CUSTOMER_TYPE", "ACTIVATED", "STATUS_NAME"]
available = [c for c in cat_cols_prime if c in prime_df.columns]

if available:
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, available):
        counts = prime_df[col].value_counts().head(8)
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=sns.color_palette("Set2", len(counts)),
               startangle=140, textprops={"fontsize": 9})
        ax.set_title(col.replace("_", " ").title())
    fig.suptitle("Customer Demographics Breakdown", fontsize=15, y=1.02)
    save(fig, "02_demographics_pie.png")

# Product distribution (bar chart)
if "PRODUCT_NAME" in prime_df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    top = prime_df["PRODUCT_NAME"].value_counts().head(15)
    sns.barplot(x=top.values, y=top.index, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Accounts")
    ax.set_title("Top 15 Credit Card Products")
    save(fig, "02_product_distribution.png")

# =====================================================================
# 4. NUMERIC DISTRIBUTIONS (before cleaning)
# =====================================================================
print("\n▶ Plotting numeric distributions …")

dist_cols = ["CREDIT_LIMIT", "LEDGER_BALANCE", "OVERDUEAMOUNT",
             "LAST_PAYMENT_AMOUNT", "JOINING_FEE", "ANNUAL_FEE"]
dist_cols = [c for c in dist_cols if c in prime_df.columns]

if dist_cols:
    rows = (len(dist_cols) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4.5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(dist_cols):
        data = prime_df[col].dropna()
        # Clip at 99th percentile to remove extreme outliers for cleaner visuals
        upper = data.quantile(0.99)
        data = data[data <= upper]
        sns.histplot(data, bins=50, kde=True, ax=axes[i], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if abs(x) >= 1000 else f"{x:.0f}"))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribution of Key Financial Variables (≤ 99th pctl)", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "03_numeric_distributions.png")

# Transaction amounts
txn_dist = ["BILLING AMT", "ORIG AMOUNT"]
txn_dist = [c for c in txn_dist if c in transaction_df.columns]
if txn_dist:
    fig, axes = plt.subplots(1, len(txn_dist), figsize=(7 * len(txn_dist), 5))
    if len(txn_dist) == 1:
        axes = [axes]
    for ax, col in zip(axes, txn_dist):
        data = transaction_df[col].dropna()
        # Clip at 99th percentile to remove extreme outliers for cleaner visuals
        upper = data.quantile(0.99)
        data = data[data <= upper]
        sns.histplot(data, bins=60, kde=True, ax=ax, color="#3498db")
        ax.set_title(f"Transaction {col}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if abs(x) >= 1000 else f"{x:.0f}"))
    fig.suptitle("Transaction Amount Distributions (≤ 99th pctl)", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "03_transaction_amounts.png")

# =====================================================================
# 5. OUTLIER ANALYSIS  (before vs after 99th-percentile cap)
# =====================================================================
print("\n▶ Plotting outlier capping comparison …")

outlier_cols_txn = [c for c in ["BILLING AMT", "ORIG AMOUNT"] if c in transaction_df.columns]
for col in outlier_cols_txn:
    p99 = transaction_df[col].quantile(0.99)
    transaction_df[col] = np.where(transaction_df[col] > p99, p99, transaction_df[col])

if "OVERDUEAMOUNT" in prime_df.columns:
    p99_od = prime_df["OVERDUEAMOUNT"].quantile(0.99)
    prime_df["OVERDUEAMOUNT"] = np.where(prime_df["OVERDUEAMOUNT"] > p99_od, p99_od, prime_df["OVERDUEAMOUNT"])

compare_cols = [("BILLING AMT", txn_raw, transaction_df),
                ("ORIG AMOUNT", txn_raw, transaction_df),
                ("OVERDUEAMOUNT", prime_raw, prime_df)]
compare_cols = [(n, r, c) for n, r, c in compare_cols if n in r.columns]

if compare_cols:
    fig, axes = plt.subplots(len(compare_cols), 2, figsize=(14, 4.5 * len(compare_cols)))
    if len(compare_cols) == 1:
        axes = axes.reshape(1, -1)
    for row, (col, raw, cleaned) in enumerate(compare_cols):
        sns.boxplot(x=raw[col].dropna(), ax=axes[row, 0], color="#e74c3c")
        axes[row, 0].set_title(f"{col} — Before Capping")
        sns.boxplot(x=cleaned[col].dropna(), ax=axes[row, 1], color="#2ecc71")
        axes[row, 1].set_title(f"{col} — After 99th-Pctl Cap")
    fig.suptitle("Outlier Treatment: Before vs After", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "04_outlier_comparison.png")

# =====================================================================
# 6. FEATURE ENGINEERING VISUALS
# =====================================================================
print("\n▶ Engineering features & plotting …")

extraction_date = pd.to_datetime("today")

# Account tenure
if "CREATION_DATE" in prime_df.columns:
    prime_df["ACCOUNT_TENURE_MONTHS"] = (extraction_date - prime_df["CREATION_DATE"]).dt.days / 30

# Days since last payment
if "LAST_PAYMENT_DATE" in prime_df.columns:
    prime_df["DAYS_SINCE_LAST_PAYMENT"] = (extraction_date - prime_df["LAST_PAYMENT_DATE"]).dt.days

# Age
if "DOB" in prime_df.columns:
    prime_df["AGE"] = (extraction_date - prime_df["DOB"]).dt.days // 365
    bins = [18, 25, 35, 50, 65, 100]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    prime_df["AGE_GROUP"] = pd.cut(prime_df["AGE"], bins=bins, labels=labels, right=True)

# Credit limit bands
if "CREDIT_LIMIT" in prime_df.columns:
    limit_bins = [0, 10000, 50000, 100000, 500000, np.inf]
    limit_labels = ["Low", "Medium", "High", "Very High", "Premium"]
    prime_df["LIMIT_BAND"] = pd.cut(prime_df["CREDIT_LIMIT"], bins=limit_bins, labels=limit_labels)

# Fee-to-limit ratio
if all(c in prime_df.columns for c in ["JOINING_FEE", "ANNUAL_FEE", "CREDIT_LIMIT"]):
    prime_df["FEE_TO_LIMIT_RATIO"] = (
        (prime_df["JOINING_FEE"] + prime_df["ANNUAL_FEE"])
        / prime_df["CREDIT_LIMIT"].replace({0: np.nan})
    )

# Card replacement freq
rep_cols = ["FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD"]
if all(c in prime_df.columns for c in rep_cols):
    prime_df["CARD_REPLACEMENT_FREQ"] = prime_df[rep_cols].sum(axis=1)

# Foreign transaction flag
if "TRXN COUNTRY" in transaction_df.columns:
    transaction_df["IS_FOREIGN_TRXN"] = (transaction_df["TRXN COUNTRY"] != "EGYPT").fillna(False).astype(int)

# ── Age Group Bar ──
if "AGE_GROUP" in prime_df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["18-25", "26-35", "36-50", "51-65", "65+"]
    counts = prime_df["AGE_GROUP"].value_counts().reindex(order).fillna(0)
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("magma", len(counts)),
                  edgecolor="white", linewidth=1.2)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                f"{int(b.get_height()):,}", ha="center", fontsize=10)
    ax.set_title("Customer Age Group Distribution")
    ax.set_ylabel("Count")
    save(fig, "05_age_groups.png")

# ── Credit Limit Bands ──
if "LIMIT_BAND" in prime_df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Low", "Medium", "High", "Very High", "Premium"]
    counts = prime_df["LIMIT_BAND"].value_counts().reindex(order).fillna(0)
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("rocket", len(counts)),
                  edgecolor="white", linewidth=1.2)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                f"{int(b.get_height()):,}", ha="center", fontsize=10)
    ax.set_title("Credit Limit Band Distribution")
    ax.set_ylabel("Count")
    save(fig, "05_limit_bands.png")

# ── Account Tenure ──
if "ACCOUNT_TENURE_MONTHS" in prime_df.columns:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(prime_df["ACCOUNT_TENURE_MONTHS"].dropna(), bins=40, kde=True,
                 color="#1abc9c", ax=ax)
    ax.set_title("Account Tenure (Months)")
    ax.set_xlabel("Months since Account Creation")
    save(fig, "05_account_tenure.png")

# ── Days Since Last Payment ──
if "DAYS_SINCE_LAST_PAYMENT" in prime_df.columns:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = prime_df["DAYS_SINCE_LAST_PAYMENT"].dropna()
    data = data[data < data.quantile(0.99)]  # trim for viz
    sns.histplot(data, bins=40, kde=True, color="#e67e22", ax=ax)
    ax.set_title("Days Since Last Payment")
    save(fig, "05_days_since_payment.png")

# =====================================================================
# 7. RFM FEATURES  (Recency, Frequency, Monetary)
# =====================================================================
print("\n▶ Plotting RFM features …")

if "RIMNO" in transaction_df.columns and "BILLING AMT" in transaction_df.columns:
    rfm = transaction_df.groupby("RIMNO").agg(
        TOTAL_SPEND=("BILLING AMT", "sum"),
        AVG_TXN=("BILLING AMT", "mean"),
        TXN_COUNT=("BILLING AMT", "count"),
        RECENCY=("TRXN DATE", lambda x: (extraction_date - x.max()).days)
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(rfm["TOTAL_SPEND"], bins=50, kde=True, ax=axes[0, 0], color="#8e44ad")
    axes[0, 0].set_title("Total Spend per Customer")

    sns.histplot(rfm["AVG_TXN"], bins=50, kde=True, ax=axes[0, 1], color="#2980b9")
    axes[0, 1].set_title("Average Transaction Amount")

    sns.histplot(rfm["TXN_COUNT"], bins=50, kde=True, ax=axes[1, 0], color="#27ae60")
    axes[1, 0].set_title("Transaction Frequency")

    sns.histplot(rfm["RECENCY"], bins=50, kde=True, ax=axes[1, 1], color="#c0392b")
    axes[1, 1].set_title("Recency (Days Since Last Txn)")

    fig.suptitle("RFM Analysis — Customer Transaction Behaviour", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "06_rfm_analysis.png")

# =====================================================================
# 8. FOREIGN vs DOMESTIC TRANSACTIONS
# =====================================================================
print("\n▶ Plotting cross-border analysis …")

if "IS_FOREIGN_TRXN" in transaction_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie
    counts = transaction_df["IS_FOREIGN_TRXN"].value_counts()
    labels_pie = ["Domestic", "Foreign"]
    axes[0].pie(counts, labels=labels_pie, autopct="%1.1f%%",
                colors=["#3498db", "#e74c3c"], startangle=90,
                explode=(0, 0.06), textprops={"fontsize": 12})
    axes[0].set_title("Domestic vs Foreign Transactions")

    # Top foreign countries
    if "TRXN COUNTRY" in transaction_df.columns:
        foreign = transaction_df[transaction_df["IS_FOREIGN_TRXN"] == 1]
        top_countries = foreign["TRXN COUNTRY"].value_counts().head(10)
        sns.barplot(x=top_countries.values, y=top_countries.index, palette="flare", ax=axes[1])
        axes[1].set_title("Top 10 Foreign Transaction Countries")
        axes[1].set_xlabel("Number of Transactions")

    fig.suptitle("Cross-Border Transaction Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, "07_foreign_vs_domestic.png")

# =====================================================================
# 9. TRANSACTION PATTERNS OVER TIME
# =====================================================================
print("\n▶ Plotting transaction time-series …")

if "TRXN DATE" in transaction_df.columns:
    # Filter to only include data from Jan 2025 to March 2026
    date_mask = (
        (transaction_df["TRXN DATE"] >= "2025-01-01") &
        (transaction_df["TRXN DATE"] <= "2026-03-31")
    )
    txn_filtered = transaction_df.loc[date_mask].copy()

    ts = txn_filtered.set_index("TRXN DATE").resample("ME")
    monthly = pd.DataFrame({
        "TXN_COUNT": ts["BILLING AMT"].count(),
        "TOTAL_SPEND": ts["BILLING AMT"].sum()
    }).dropna()

    if not monthly.empty:
        fig, ax1 = plt.subplots(figsize=(14, 5))
        color1, color2 = "#2980b9", "#e74c3c"
        ax1.fill_between(monthly.index, monthly["TXN_COUNT"], alpha=0.25, color=color1)
        ax1.plot(monthly.index, monthly["TXN_COUNT"], color=color1, linewidth=2, label="Txn Count", marker="o")
        ax1.set_ylabel("Transaction Count", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(monthly.index, monthly["TOTAL_SPEND"], color=color2, linewidth=2, label="Total Spend", marker="s")
        ax2.set_ylabel("Total Spend (EGP)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        # Format x-axis to show monthly labels nicely
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax1.set_title("Monthly Transaction Volume & Spend (Jan 2025 – Mar 2026)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        save(fig, "08_monthly_trends.png")

# =====================================================================
# 10. MCC CATEGORY SPEND BREAKDOWN
# =====================================================================
print("\n▶ Plotting MCC spend breakdown …")

if "MCC" in transaction_df.columns and "BILLING AMT" in transaction_df.columns:
    mcc_totals = transaction_df.groupby("MCC")["BILLING AMT"].sum().sort_values(ascending=False).head(15)
    
    # Define your MCC mapping here (or import it if it's in another file)
    MCC_MAPPING = {
        4814: "Telecommunication Services",
        4900: "Utilities",
        5065: "Electrical Parts & Equipment",
        4121: "Taxicabs and Limousines",
        5331: "Variety Stores",
        5411: "Grocery Stores & Supermarkets",
        5499: "Convenience Stores",
        5541: "Service Stations / Gas",
        5732: "Electronic Sales",
        5814: "Fast Food Restaurants",
        6011: "Automated Cash Disbursements",
        7278: "Buying/Shopping Clubs",
        8099: "Medical Services",
        8299: "Educational Services",
        9222: "Fines",
        9399: "Government Services",
    }
    
    if not mcc_totals.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        
        # Map the MCC codes to categories. If not found in mapping, default to the MCC number.
        # Handle cases where mcc might be a float, int, or string
        mcc_labels = [MCC_MAPPING.get(int(float(mcc)), str(mcc)) for mcc in mcc_totals.index]
        
        sns.barplot(x=mcc_totals.values, y=mcc_labels, palette="mako", ax=ax)
        ax.set_title("Top 15 Merchant Category Codes by Total Spend")
        ax.set_xlabel("Total Billing Amount (EGP)")
        save(fig, "09_mcc_spend.png")

# =====================================================================
# 11. CORRELATION HEATMAP (engineered features)
# =====================================================================
print("\n▶ Plotting correlation heatmap …")

corr_cols = ["CREDIT_LIMIT", "LEDGER_BALANCE", "OVERDUEAMOUNT",
             "LAST_PAYMENT_AMOUNT", "DELIQUENCY", "NO_OF_CYCLES",
             "ACCOUNT_TENURE_MONTHS", "DAYS_SINCE_LAST_PAYMENT",
             "AGE", "FEE_TO_LIMIT_RATIO", "CARD_REPLACEMENT_FREQ"]
corr_cols = [c for c in corr_cols if c in prime_df.columns]

if len(corr_cols) >= 3:
    corr = prime_df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15)
    save(fig, "10_correlation_heatmap.png")

# =====================================================================
# 12. DELINQUENCY & RISK INDICATORS
# =====================================================================
print("\n▶ Plotting delinquency / risk indicators …")

if "DELIQUENCY" in prime_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    del_counts = prime_df["DELIQUENCY"].value_counts().sort_index().head(10)
    axes[0].bar(del_counts.index.astype(str), del_counts.values,
                color=sns.color_palette("Reds_d", len(del_counts)), edgecolor="white")
    axes[0].set_title("Delinquency Cycle Distribution")
    axes[0].set_xlabel("Delinquency Cycles")
    axes[0].set_ylabel("Number of Accounts")

    if "OVERDUEAMOUNT" in prime_df.columns:
        delinquent = prime_df[prime_df["DELIQUENCY"] > 0]
        if not delinquent.empty:
            sns.boxplot(x="DELIQUENCY", y="OVERDUEAMOUNT", data=delinquent.head(5000),
                        palette="Reds", ax=axes[1])
            axes[1].set_title("Overdue Amount by Delinquency Level")

    fig.suptitle("Credit Risk Indicators", fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, "11_delinquency_risk.png")

# =====================================================================
# 13. BRANCH-LEVEL ANALYSIS
# =====================================================================
print("\n▶ Plotting branch analysis …")

if "BRANCH_NAME" in prime_df.columns:
    top_branches = prime_df["BRANCH_NAME"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=top_branches.values, y=top_branches.index, palette="crest", ax=ax)
    ax.set_title("Top 15 Branches by Number of Accounts")
    ax.set_xlabel("Account Count")
    save(fig, "12_branch_distribution.png")

# =====================================================================
# DONE
# =====================================================================
print(f"\n{'='*55}")
print(f"  All plots saved to ./{OUT}/")
print(f"  Total plots generated: {len(os.listdir(OUT))}")
print(f"{'='*55}")
