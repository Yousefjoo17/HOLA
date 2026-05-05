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

final_columns_to_drop = ['AGE', "BRANCH_NAME","PRODUCT_NAME","DOB","RIMNO","DOB_WAS_MISSING"]
existing_cols_to_drop = [col for col in final_columns_to_drop if col in final_customer_profile.columns]
final_customer_profile = final_customer_profile.drop(columns=existing_cols_to_drop)

# One-hot encode AGE_GROUP
age_group_dummies = pd.get_dummies(final_customer_profile['AGE_GROUP'], prefix='AGE_GROUP', drop_first=False)
age_group_dummies.columns = [col.replace('-', '_') for col in age_group_dummies.columns]
final_customer_profile = pd.concat([final_customer_profile, age_group_dummies], axis=1)
final_customer_profile = final_customer_profile.drop(columns=['AGE_GROUP'])

# One-hot encode GENDER
gender_dummies = pd.get_dummies(final_customer_profile['GENDER'], prefix='GENDER', drop_first=False)
final_customer_profile = pd.concat([final_customer_profile, gender_dummies], axis=1)
final_customer_profile = final_customer_profile.drop(columns=['GENDER'])

final_customer_profile.to_csv("final_customer_profile.csv", index=False)


# I need to make analytcis and see correlation with products to help me understand the full resultant data final_customer_profile 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. Load the Final Data
# =========================================================
print("Loading final_customer_profile.csv...")
df = pd.read_csv("final_customer_profile.csv")

import pandas as pd

df = pd.read_csv("final_customer_profile.csv")

for col in df.columns:
    print(col, df[col].dtype)

# Identify product columns dynamically
prod_cols = [col for col in df.columns if col.startswith('HAS_PROD_')]

# Identify numeric columns for correlation (excluding ID)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'CUSTOMER_ID' in numeric_cols:
    numeric_cols.remove('CUSTOMER_ID')

# Set plot style
sns.set_theme(style="whitegrid")

# =========================================================
# 2. Product Ownership Distribution
# =========================================================
plt.figure(figsize=(10, 6))
prod_counts = df[prod_cols].sum().sort_values(ascending=False)
sns.barplot(x=prod_counts.values, y=prod_counts.index, palette='viridis')

plt.title('Total Customers per Credit Card Product', fontsize=14)
plt.xlabel('Number of Customers')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('1_product_distribution.png')
plt.show()

# =========================================================
# 3. Overall Feature Correlation Heatmap
# =========================================================
plt.figure(figsize=(16, 12))
# Calculate Pearson correlation
corr_matrix = df[numeric_cols].corr()

# Create a mask to show only the lower half of the heatmap (less cluttered)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .7})

plt.title('Correlation Heatmap: Demographics, RFM, and Products', fontsize=16)
plt.tight_layout()
plt.savefig('2_correlation_heatmap.png')
plt.show()

# =========================================================
# 4. Correlation Specifically with Products
# =========================================================
# Let's isolate how RFM and Spends correlate with specific products
print("\n--- Top Correlations with Products ---")
for prod in prod_cols:
    if prod in corr_matrix.columns:
        print(f"\nTop correlated features for {prod}:")
        # Get correlations for this product, drop the product itself, and sort
        corrs = corr_matrix[prod].drop(prod).sort_values(ascending=False)
        # Show top 5 positive and top 3 negative
        print(pd.concat([corrs.head(5), corrs.tail(3)]))

# =========================================================
# 5. Spend Behavior by Demographics (Age & Gender)
# =========================================================
plt.figure(figsize=(12, 6))
# Order the age groups logically
age_order = ['18-25', '26-35', '36-50', '51-65', '65+']

sns.barplot(data=df, x='AGE_GROUP', y='TOTAL_SPEND_AMT', hue='GENDER', 
            order=age_order, errorbar=None, palette='Set2')

plt.title('Average Total Spend by Age Group and Gender', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Average Total Spend (Billing Amt)')
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('3_spend_by_demographics.png')
plt.show()

# =========================================================
# 6. RFM Scatter: Frequency vs Monetary
# =========================================================
plt.figure(figsize=(10, 6))

# We use log scale because spend and transactions usually have massive outliers
sns.scatterplot(data=df, x='TRXN_COUNT', y='TOTAL_SPEND_AMT', 
                hue='FOREIGN_TRXN_COUNT', size='FOREIGN_TRXN_COUNT', 
                sizes=(20, 200), palette='flare', alpha=0.7)

plt.title('Transaction Count vs Total Spend (Colored by Foreign Trxns)', fontsize=14)
plt.xlabel('Total Transaction Count')
plt.ylabel('Total Spend Amount')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('4_rfm_scatter.png')
plt.show()