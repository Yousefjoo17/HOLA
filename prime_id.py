import pandas as pd
import numpy as np
import glob
import os


# ========================= Helper Functions (from main.py) =========================
def apply_cast_and_report(df, columns, cast_type):
    # Get the total number of rows for our percentage math
    total_rows = len(df)
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        # Count nulls before
        nulls_before = df[col].isna().sum()
        
        # Apply the specific vectorized casting logic
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
            
        # Count nulls after
        nulls_after = df[col].isna().sum()
        
        # Calculate percentage of missing data
        pct_missing = (nulls_after / total_rows) * 100
        
        # Print the results, adding the percentage formatted to 2 decimal places
        print(f"[{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}% missing)")


# ========================= Column Definitions (from main.py) =========================
prime_string_cols = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
    "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "
]
prime_int_cols = ["BRANCH_ID", "RIMNO"]
prime_float_cols = [
    "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD", "SETTLEMENT AMT"
]
prime_date_cols = ["CREATION_DATE", "LAST_STAEMENT_DATE", "LAST_PAYMENT_DATE", "DOB", "CLOSURE_DATE"]


# ========================= Output Directory =========================
output_dir = "prime_cleaned"
os.makedirs(output_dir, exist_ok=True)


# ========================= Find All Prime Files =========================
prime_files = glob.glob("prime/*.csv")

if not prime_files:
    print("ERROR: No CSV files found in 'prime/' folder. Make sure the folder exists.")
    exit(1)

print(f"Found {len(prime_files)} prime file(s).\n")


# ========================= Process Each File Separately =========================
for file in prime_files:
    file_basename = os.path.splitext(os.path.basename(file))[0]
    print(f"{'='*60}")
    print(f"Processing: {file}")
    print(f"{'='*60}")

    # --- Load ---
    df = pd.read_csv(
        file,
        encoding='latin',
        dtype={col: "string" for col in prime_string_cols + prime_int_cols + prime_float_cols},
        parse_dates=prime_date_cols
    ).rename(columns={'RIM_NO': 'RIMNO', 'NAME': 'PRODUCT_NAME'})

    print(f"  Rows loaded: {len(df)}")

    # --- Cast all columns using helper function ---
    print("\n  -> String Columns:")
    apply_cast_and_report(df, prime_string_cols, 'string')

    print("\n  -> Float Columns:")
    apply_cast_and_report(df, prime_float_cols, 'float')

    print("\n  -> Integer Columns:")
    apply_cast_and_report(df, prime_int_cols, 'int')

    print("\n  -> Date Columns:")
    apply_cast_and_report(df, prime_date_cols, 'date')

    # Clean up ACTIVATED and STATUS for reliable splitting
    df['ACTIVATED'] = df['ACTIVATED'].astype("string").str.strip().str.upper()
    df['STATUS'] = df['STATUS'].astype("string").str.strip().str.upper()

    # Status codes that indicate an inactive / closed / blocked card
    inactive_statuses = [
        'CLSB',   # closed by bank
        'CLSC',   # closed by customer
        'LOST',   # lost card
        'WROF',   # write off status
        'CNCD',   # card not collected by DSU
        'SUSP',   # profit suspended
        'FRAD',   # pick up card special fraud
        'CLSD',   # closed
        'BLOK',   # blok card
        'NOAU',   # no authorization
        'EXMU',   # expired murabha
        'PICK',   # pick up card
        'BLCK',   # blocked status
        'STLC',   # stolen card
        'OFBL',   # offline pin block
        'ONBL',   # online pin block
        'FREZ',   # freeze card
        'EXPD',   # expired
        'EXPC',   # expired card
    ]

    # --- Assign CUSTOMER_ID based on unique (RIMNO, DOB) pairs in THIS file ---
    unique_pairs = df[['RIMNO', 'DOB']].drop_duplicates().reset_index(drop=True)
    unique_pairs['CUSTOMER_ID'] = range(1, len(unique_pairs) + 1)

    print(f"  Unique (RIMNO, DOB) pairs: {len(unique_pairs)}")

    df = df.merge(unique_pairs, on=['RIMNO', 'DOB'], how='left')

    unmatched = df['CUSTOMER_ID'].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} rows could not be assigned a CUSTOMER_ID (missing RIMNO or DOB).")

    # --- Split into Active & Historical ---
    # Historical: ACTIVATED is D/I  OR  STATUS is an inactive code
    is_inactive_activated = df['ACTIVATED'].isin(['D', 'I'])
    is_inactive_status = df['STATUS'].isin(inactive_statuses)
    is_historical = is_inactive_activated | is_inactive_status

    historical_df = df[is_historical].copy()
    active_df = df[(df['ACTIVATED'] == 'A') & ~is_inactive_status].copy()
    other_df = df[~is_historical & ~((df['ACTIVATED'] == 'A') & ~is_inactive_status)]

    print(f"  Active   (ACTIVATED='A' & active STATUS):   {len(active_df)} rows")
    print(f"  Historical (ACTIVATED='D'/'I' or inactive STATUS): {len(historical_df)} rows")
    if len(other_df) > 0:
        print(f"  Other (unrecognized): {len(other_df)} rows")
        print(f"    ACTIVATED values: {other_df['ACTIVATED'].unique().tolist()}")
        print(f"    STATUS values:    {other_df['STATUS'].unique().tolist()}")

    # --- Save ---
    active_path = os.path.join(output_dir, f"{file_basename}_active.csv")
    historical_path = os.path.join(output_dir, f"{file_basename}_historical.csv")

    active_df.to_csv(active_path, index=False)
    historical_df.to_csv(historical_path, index=False)

    print(f"  Saved -> {active_path}")
    print(f"  Saved -> {historical_path}")

    if len(other_df) > 0:
        other_path = os.path.join(output_dir, f"{file_basename}_other.csv")
        other_df.to_csv(other_path, index=False)
        print(f"  Saved -> {other_path}")

    print()

print("All files processed!")
