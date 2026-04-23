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

