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
prime_date_cols = ["CREATION_DATE", "LAST_STATEMENT_DATE", "LAST_PAYMENT_DATE", "DOB", "CLOSURE_DATE"]

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
                        'SECOND_REPLACED_CARD','THIRD_REPLACED_CARD', 'LAST_STATEMENT_DATE', "LEDGER_BALANCE", "AVAILABLE_LIMIT",
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

transaction_df['IS_FOREIGN_TRXN'] = (transaction_df['TRXN COUNTRY'] != 'EGYPT').fillna(False).astype(int)

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


# One-hot encode AGE_GROUP
age_group_dummies = pd.get_dummies(final_customer_profile['AGE_GROUP'], prefix='AGE_GROUP', drop_first=False)
age_group_dummies.columns = [col.replace('-', '_') for col in age_group_dummies.columns]
final_customer_profile = pd.concat([final_customer_profile, age_group_dummies], axis=1)
final_customer_profile = final_customer_profile.drop(columns=['AGE_GROUP'])

# One-hot encode GENDER
gender_dummies = pd.get_dummies(final_customer_profile['GENDER'], prefix='GENDER', drop_first=False)
final_customer_profile = pd.concat([final_customer_profile, gender_dummies], axis=1)
final_customer_profile = final_customer_profile.drop(columns=['GENDER'])


final_columns_to_drop = ['AGE', "BRANCH_NAME","PRODUCT_NAME","DOB","RIMNO","DOB_WAS_MISSING", "GENDER_Unknown"]
existing_cols_to_drop = [col for col in final_columns_to_drop if col in final_customer_profile.columns]
final_customer_profile = final_customer_profile.drop(columns=existing_cols_to_drop)

final_customer_profile.to_csv("final_customer_profile.csv", index=False)


# I need to make analytcis and see correlation with products to help me understand the full resultant data final_customer_profile 

import pandas as pd
print("Loading final_customer_profile.csv...")
final_customer_profile = pd.read_csv("final_customer_profile.csv")

import os
import matplotlib.pyplot as plt
import seaborn as sns

# ========================= 7. Product Correlation & Analytics =========================
print("\n============================================================")
print(" PHASE 4: Product Analytics & Correlation")
print("============================================================\n")

# --- PATH CONFIGURATION ---
# If running as a standard .py script, this finds the script's exact folder:
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if you are running this inside a Jupyter Notebook 
    # (Jupyter doesn't support __file__)
    script_dir = os.getcwd()

# Define explicit paths for your output files
heatmap_path = os.path.join(script_dir, 'product_feature_correlation_heatmap.png')
summary_path = os.path.join(script_dir, 'product_profile_summary.csv')
# --------------------------

# 1. Identify Product Columns vs Feature Columns
product_cols = [col for col in final_customer_profile.columns if col.startswith('HAS_PROD_')]
exclude_cols = ['CUSTOMER_ID', 'BRANCH_ID'] + product_cols
feature_cols = [col for col in final_customer_profile.columns if col not in exclude_cols]

# 2. Calculate Correlation Matrix
print("Calculating feature correlations...")
corr_matrix = final_customer_profile[feature_cols + product_cols].corr()
prod_corr = corr_matrix.loc[feature_cols, product_cols]

# 3. Print Top Correlated Features per Product
print("\n--- Top Driving Features per Product ---")
for prod in product_cols:
    product_name = prod.replace('HAS_PROD_', '')
    print(f"\n>> Product: {product_name}")
    
    sorted_corrs = prod_corr[prod].dropna().sort_values(ascending=False)
    
    print("  Top Positive Drivers:")
    print(sorted_corrs.head(3).to_string())
    print("\n  Top Negative Drivers:")
    print(sorted_corrs.tail(3).sort_values(ascending=True).to_string())

# 4. Generate & Save a Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(prod_corr, annot=False, cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Pearson Correlation'})

plt.title('Correlation: Customer Behaviors & Demographics vs. Product Ownership', fontsize=14)
plt.ylabel('Customer Features', fontsize=12)
plt.xlabel('Products', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save using the explicitly mapped directory path
plt.savefig(heatmap_path, dpi=300)
print(f"\n[Saved] Correlation heatmap saved exactly to: \n  -> {heatmap_path}")
plt.close()

# 5. Product Level Summary (Averages)
print("\n--- Average Customer Profile per Product ---")
key_metrics = ['TOTAL_SPEND_AMT', 'AVG_TRXN_AMT', 'TRXN_COUNT', 'DAYS_SINCE_LAST_TRXN', 'FOREIGN_TRXN_COUNT']
safe_metrics = [m for m in key_metrics if m in final_customer_profile.columns]

summary_list = []
for prod in product_cols:
    subset = final_customer_profile[final_customer_profile[prod] == 1]
    if len(subset) > 0:
        avg_metrics = subset[safe_metrics].mean().to_dict()
        avg_metrics['Product'] = prod.replace('HAS_PROD_', '')
        avg_metrics['Total_Customers'] = len(subset)
        summary_list.append(avg_metrics)

if summary_list:
    summary_df = pd.DataFrame(summary_list)
    cols = ['Product', 'Total_Customers'] + safe_metrics
    summary_df = summary_df[cols]
    
    print(summary_df.to_string(index=False))
    
    # Save using the explicitly mapped directory path
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Saved] Product summary saved exactly to: \n  -> {summary_path}")



    # ========================= 8. Population Reconcilliation =========================

# 1. Count how many total products each customer has
product_cols = [col for col in final_customer_profile.columns if col.startswith('HAS_PROD_')]
products_per_customer = final_customer_profile[product_cols].sum(axis=1)

# 2. Calculate the splits
total_customers = len(final_customer_profile)
no_product_count = (products_per_customer == 0).sum()
has_product_count = (products_per_customer > 0).sum()

# 3. Print the breakdown BEFORE the summary concept
print("\n============================================================")
print(" POPULATION BREAKDOWN")
print("============================================================")
print(f"Total Unique Customers in Data : {total_customers}")
print(f"Customers with NO products     : {no_product_count}")
print(f"Customers with 1+ products     : {has_product_count}")
print("-" * 60)
print(f"Math Check: {no_product_count} + {has_product_count} = {no_product_count + has_product_count}")

# (Assuming summary_df is already created from the previous code)
print("\n--- Product Summary Table ---")
print(summary_df[['Product', 'Total_Customers']].to_string(index=False))

# 4. Print the breakdown AFTER the summary concept
sum_of_summary = summary_df['Total_Customers'].sum()

print("\n============================================================")
print(" POST-SUMMARY RECONCILIATION")
print("============================================================")
print(f"Sum of customers in the Summary Table: {sum_of_summary}")
print(f"Unique customers with 1+ products    : {has_product_count}")
print("-" * 60)
print(f"Difference (Overlap)                 : {sum_of_summary - has_product_count}")

print("\n💡 What this means:")
print("If the sum of the Summary Table is LARGER than your unique customers with products,")
print("it means you have cross-sell overlap! Specifically, there are " + str(sum_of_summary - has_product_count) + " instances")
print("where a customer owns more than one product and is being counted in multiple buckets.")