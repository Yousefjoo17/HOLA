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
    prime_df = pd.concat(prime_dfs, ignore_index=True)
    
    print(f"Loaded {len(prime_df)} total rows from {len(prime_files)} files.\n")
    print("Applying Casting to Master Prime Data:")
    
    apply_cast_and_report(prime_df, prime_string_cols, 'string')
    apply_cast_and_report(prime_df, prime_int_cols, 'int')
    apply_cast_and_report(prime_df, prime_float_cols, 'float')
    apply_cast_and_report(prime_df, prime_date_cols, 'date')


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
    transaction_df = pd.concat(txn_dfs, ignore_index=True)
    
    print(f"Loaded {len(transaction_df)} total rows from {len(txn_files)} files.\n")
    print("Applying Casting to Master Transaction Data:")
    
    apply_cast_and_report(transaction_df, transaction_string_cols, 'string')
    apply_cast_and_report(transaction_df, transaction_int_cols, 'int')
    apply_cast_and_report(transaction_df, transaction_float_cols, 'float')
    apply_cast_and_report(transaction_df, transaction_date_cols, 'date')

print("All consolidations complete!")

# ========================= 3. Cleanup & Final Info ==========================
prime_columns_to_drop = ["MAPPING_ACCNO", "STATUS","CREATION_DATE","MIN_PAYMENT", "OVER_LIMIT", "ACTIVATED",
                        "DELIQUENCY","STATUS_NAME","OVER_LIMIT","TOTAL_HOLD" ,"ORGANIZATION","JOINING_FEE",
                        "ANNUAL_FEE","LAST_PAYMENT_AMOUNT", "LAST_PAYMENT_DATE", "SETTLEMENT AMT",'FIRST_REPLACED_CARD',
                        'SECOND_REPLACED_CARD','THIRD_REPLACED_CARD', 'LAST_STAEMENT_DATE', "LEDGER_BALANCE", "AVAILABLE_LIMIT",
                        "CLOSURE_DATE", "Card account status ", "CUSTOMER_TYPE", "OVERDUEAMOUNT"]
existing_cols_to_drop = [col for col in prime_columns_to_drop if col in prime_df.columns]
prime_df = prime_df.drop(columns=existing_cols_to_drop)

print("\n--- Final Dataframe Info ---")
print(prime_df.info())



transaction_columns_to_drop = ["POST DATE","ORIG AMOUNT","EMBEDDED _FEE", "SETTLEMENT AMT", "SETTLEMENT CCY", "SOURCES"]
existing_cols_to_drop = [col for col in transaction_columns_to_drop if col in transaction_df.columns]
transaction_df = transaction_df.drop(columns=existing_cols_to_drop)

print("\n--- Final Transaction Dataframe Info ---")
print(transaction_df.info())

prime_df["GENDER"] = prime_df["GENDER"].fillna("Unknown")





#=========================transaction Encoding ==========================
trxn_categorical_cols = ["PRODUCT_NAME","MCC","BANKBRANCH"]
for col in trxn_categorical_cols:
    unique_values = transaction_df[col].unique()
    print(f"Unique values in {col}: {unique_values}")

#=====================Prime Encoding=========================
prime_categorical_cols = ["PRODUCT_NAME","BRANCH_NAME"]
for col in prime_categorical_cols:
    unique_values = prime_df[col].unique()
    print(f"Unique values in {col}: {unique_values}")


    
Product_frequency_threshold = 500

prime_counts = prime_df["PRODUCT_NAME"].value_counts()
prime_df["PRODUCT_NAME"] = prime_df["PRODUCT_NAME"].map(
    lambda x: x if prime_counts.get(x, 0) >= Product_frequency_threshold else "another product"
)

transaction_counts = transaction_df["PRODUCT_NAME"].value_counts()
transaction_df["PRODUCT_NAME"] = transaction_df["PRODUCT_NAME"].map(
    lambda x: x if transaction_counts.get(x, 0) >= Product_frequency_threshold else "another product"
)

MCC_frequency_threshold = 5000
transaction_counts = transaction_df["MCC"].value_counts()
transaction_df["MCC"] = transaction_df["MCC"].map(
    lambda x: x if transaction_counts.get(x, 0) >= MCC_frequency_threshold else "another product"
)


user_item_matrix = pd.crosstab(prime_df['CUSTOMER_ID'], prime_df['PRODUCT_NAME'])
user_item_matrix.columns = [f"HAS_PROD_{col.strip().replace(' ', '_')}" for col in user_item_matrix.columns]
prime_df = prime_df.drop_duplicates(subset=['CUSTOMER_ID']).merge(user_item_matrix, on='CUSTOMER_ID', how='inner')


# 1. Standard RFM (Recency, Frequency, Monetary)
rfm_features = transaction_df.groupby('CUSTOMER_ID').agg(
    TOTAL_SPEND_AMT=('BILLING AMT', 'sum'),
    AVG_TRXN_AMT=('BILLING AMT', 'mean'),
    TRXN_COUNT=('BILLING AMT', 'count'),
    DAYS_SINCE_LAST_TRXN=('TRXN DATE', lambda x: (pd.to_datetime('today') - x.max()).days)
).reset_index()

# 2. Spend by Merchant Category Code (MCC) or Description
# This tells you if they are a traveler, foodie, online shopper, etc.
# We pivot to get the total spend per MCC per customer
mcc_spend = pd.pivot_table(
    transaction_df, 
    values='BILLING AMT', 
    index='CUSTOMER_ID', 
    columns='MCC', 
    aggfunc='sum', 
    fill_value=0
)

mcc_spend.columns = [f"MCC_{str(col)}_SPEND" for col in mcc_spend.columns]
mcc_spend = mcc_spend.reset_index()

transaction_df['IS_FOREIGN_TRXN'] = (transaction_df['TRXN COUNTRY'] != 'EG').fillna(False).astype(int)

extraction_date = pd.to_datetime("today")
prime_df['AGE'] = (extraction_date - prime_df['DOB']).dt.days // 365

bins = [18, 25, 35, 50, 65, 100]
labels = ['18-25', '26-35', '36-50', '51-65', '65+']
prime_df['AGE_GROUP'] = pd.cut(prime_df['AGE'], bins=bins, labels=labels, right=True)



# ========================= 4. Aggregate Remaining Features =========================
# You created the 'IS_FOREIGN_TRXN' flag, so let's count how many foreign 
# transactions each customer made before merging.
foreign_agg = transaction_df.groupby('CUSTOMER_ID').agg(
    FOREIGN_TRXN_COUNT=('IS_FOREIGN_TRXN', 'sum')
).reset_index()


# ========================= 5. The Final Customer 360 Merge =========================
print("\n============================================================")
print(" PHASE 3: Building Final Customer Profile (1 Row per Customer)")
print("============================================================\n")

# Your prime_df was already deduplicated and merged with the user_item_matrix
# earlier in your script. So it is the perfect base table.
final_customer_profile = prime_df.copy()

# 1. Merge RFM Features
final_customer_profile = final_customer_profile.merge(rfm_features, on='CUSTOMER_ID', how='inner')

# 2. Merge MCC Spend Matrix
final_customer_profile = final_customer_profile.merge(mcc_spend, on='CUSTOMER_ID', how='inner')

# 3. Merge Foreign Transaction Counts
final_customer_profile = final_customer_profile.merge(foreign_agg, on='CUSTOMER_ID', how='left')


# ========================= 6. Handle Missing Values (Imputation) =========================
# Because we did a LEFT JOIN, Prime customers who have NO transactions will 
# have NaN for their RFM and Spend columns. We need to fill those with 0.

# Get all MCC columns dynamically
mcc_cols = [col for col in mcc_spend.columns if col != 'CUSTOMER_ID']

# List of all transaction-based columns that should be 0 if missing
fill_zero_cols = ['TOTAL_SPEND_AMT', 'AVG_TRXN_AMT', 'TRXN_COUNT', 'FOREIGN_TRXN_COUNT'] + mcc_cols

final_customer_profile[fill_zero_cols] = final_customer_profile[fill_zero_cols].fillna(0)

# For DAYS_SINCE_LAST_TRXN, 0 would mean they transacted today. 
# Since they NEVER transacted, we should fill it with a completely out-of-range number 
# (like 9999) so your machine learning model recognizes them as inactive.
final_customer_profile['DAYS_SINCE_LAST_TRXN'] = final_customer_profile['DAYS_SINCE_LAST_TRXN'].fillna(9999)

final_columns_to_drop = ['AGE', "BRANCH_NAME","PRODUCT_NAME","DOB","RIMNO","DOB_WAS_MISSING"]
existing_cols_to_drop = [col for col in final_columns_to_drop if col in final_customer_profile.columns]
final_customer_profile = final_customer_profile.drop(columns=existing_cols_to_drop)



from sklearn.preprocessing import StandardScaler
# ========================= 6.5 Data Scaling (StandardScaler) =========================
print("Applying StandardScaler to continuous numeric features...")

# 1. Identify all numeric columns
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'Int64', 'Float64']
all_numeric_cols = final_customer_profile.select_dtypes(include=numeric_dtypes).columns.tolist()

# 2. Define columns that should NOT be scaled
exclude_from_scaling = ['CUSTOMER_ID', 'BRANCH_ID']

# Also exclude the binary columns (0/1) created by the user_item_matrix crosstab
binary_prod_cols = [col for col in all_numeric_cols if col.startswith('HAS_PROD_')]
cols_to_exclude = set(exclude_from_scaling + binary_prod_cols)

# 3. Filter down to only the columns that actually need scaling
cols_to_scale = [col for col in all_numeric_cols if col not in cols_to_exclude]

if len(cols_to_scale) > 0:
    # 4. Apply the scaler
    scaler = StandardScaler()
    final_customer_profile[cols_to_scale] = scaler.fit_transform(final_customer_profile[cols_to_scale])
    print(f"  -> Scaled {len(cols_to_scale)} continuous features.")
else:
    print("  -> No continuous features found to scale.")

    
# ========================= 7. Final Output Verification =========================
print("--- Final Dataset Dimensions ---")
print(f"Total Rows (Unique Customers): {len(final_customer_profile)}")
print(f"Total Columns (Features): {len(final_customer_profile.columns)}")

# Save the master dataset for your ML models
final_output_path = "CUSTOMER_360_FEATURES.csv"
final_customer_profile.to_csv(final_output_path, index=False)
print(f"\n✅ Saved Final Machine Learning Dataset -> {final_output_path}")

# Display a quick preview
print("\nPreview of final data:")
print(final_customer_profile[['CUSTOMER_ID', 'AGE_GROUP', 'TOTAL_SPEND_AMT', 'TRXN_COUNT']].head())


