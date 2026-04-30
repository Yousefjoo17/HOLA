import pandas as pd
import numpy as np
import glob
import os

# ========================= Helper Functions =========================
def apply_cast_and_report(df, columns, cast_type):
    """Casts columns to specific types and prints a missing data report."""
    total_rows = len(df)
    
    for col in columns:
        if col not in df.columns:
            print(f"  [Skip] Column '{col}' not found in dataframe.")
            continue
            
        nulls_before = df[col].isna().sum()
        
        if cast_type == 'string':
            df[col] = df[col].astype("string")
            
        elif cast_type == 'float':
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        elif cast_type == 'int':
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
        elif cast_type == 'date':
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        nulls_after = df[col].isna().sum()
        pct_missing = (nulls_after / total_rows) * 100
        print(f"  [{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}%)")

# ========================= Column Definitions =========================
prime_string_cols = ["BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME", "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "]
# Added CUSTOMER_ID so it gets typed properly
prime_int_cols = ["BRANCH_ID", "RIMNO", "CUSTOMER_ID"] 
prime_float_cols = ["CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT", "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD", "SETTLEMENT AMT"]
prime_date_cols = ["CREATION_DATE", "LAST_STAEMENT_DATE", "LAST_PAYMENT_DATE", "DOB", "CLOSURE_DATE"]

transaction_string_cols = ["DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES", "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME"]
# Added CUSTOMER_ID so it gets typed properly
transaction_int_cols = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY", "CUSTOMER_ID"] 
transaction_float_cols = ["ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]
transaction_date_cols = ["TRXN DATE", "POST DATE"]

# ========================= 1. Consolidate Active Prime =========================
print("============================================================")
print(" PHASE 1: Consolidating Active Prime Files")
print("============================================================\n")

prime_files = glob.glob("prime_cleaned/*_active.csv")

if not prime_files:
    print("Warning: No active prime files found in 'prime_cleaned/'.")
else:
    # Read all files as strings first to prevent mixed-type warnings during concat
    prime_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in prime_files]
    master_prime_df = pd.concat(prime_dfs, ignore_index=True)
    
    print(f"Loaded {len(master_prime_df)} total rows from {len(prime_files)} files.\n")
    print("Applying Casting to Master Prime Data:")
    
    apply_cast_and_report(master_prime_df, prime_string_cols, 'string')
    apply_cast_and_report(master_prime_df, prime_int_cols, 'int')
    apply_cast_and_report(master_prime_df, prime_float_cols, 'float')
    apply_cast_and_report(master_prime_df, prime_date_cols, 'date')

    print(f"\n✅ Saved Master Prime File -> {master_prime_df.info()}\n")
    # Save to current directory
    prime_output = "MASTER_active_prime.csv"
    master_prime_df.to_csv(prime_output, index=False)
    print(f"\n✅ Saved Master Prime File -> {prime_output}\n")


# ========================= 2. Consolidate Transactions =========================
print("============================================================")
print(" PHASE 2: Consolidating Matched Transaction Files")
print("============================================================\n")

# Grab all CSVs but EXCLUDE the missing_id files
all_txn_files = glob.glob("transaction_cleaned/*.csv")
txn_files = [f for f in all_txn_files if not f.endswith("_missing_id.csv")]

if not txn_files:
    print("Warning: No successfully matched transaction files found in 'transaction_cleaned/'.")
else:
    # Read all files as strings first
    txn_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in txn_files]
    master_txn_df = pd.concat(txn_dfs, ignore_index=True)
    
    print(f"Loaded {len(master_txn_df)} total rows from {len(txn_files)} files.\n")
    print("Applying Casting to Master Transaction Data:")
    
    apply_cast_and_report(master_txn_df, transaction_string_cols, 'string')
    apply_cast_and_report(master_txn_df, transaction_int_cols, 'int')
    apply_cast_and_report(master_txn_df, transaction_float_cols, 'float')
    apply_cast_and_report(master_txn_df, transaction_date_cols, 'date')

    print(f"\n✅ Saved Master Prime File -> {master_txn_df.info()}\n")
    # Save to current directory
    txn_output = "MASTER_transactions.csv"
    master_txn_df.to_csv(txn_output, index=False)
    print(f"\n✅ Saved Master Transaction File -> {txn_output}\n")

print("All consolidations complete!")




import pandas as pd

# 2. Perform the INNER JOIN
# We use suffixes to rename columns that exist in BOTH tables (like RIMNO)
# so you know which file they came from.
final_merged_df = pd.merge(
    master_prime_df, 
    master_txn_df, 
    on='CUSTOMER_ID', 
    how='inner',
    suffixes=('_prime', '_txn')
)

# 3. Check the results
print(f"Prime accounts: {len(master_prime_df)}")
print(f"Total transactions: {len(master_txn_df)}")
print(f"Successfully matched rows: {len(final_merged_df)}")

# 4. Save the final result
final_merged_df.to_csv("FINAL_Prime_with_Transactions.csv", index=False)
print("Saved final merged dataset.")