import pandas as pd 
import numpy as np
from datetime import datetime
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
import glob
import pandas as pd
from helpers import *

prime_string_cols = ["BRANCH_NAME","ACTIVATED","STATUS","STATUS_NAME","PRODUCT_NAME","GENDER","CUSTOMER_TYPE","Card account status "]
prime_int_cols = ["BRANCH_ID","RIMNO"]
prime_float_cols = ["CREDIT_LIMIT","DELIQUENCY","LEDGER_BALANCE","AVAILABLE_LIMIT","OVERDUEAMOUNT","FIRST_REPLACED_CARD","SECOND_REPLACED_CARD","THIRD_REPLACED_CARD","SETTLEMENT AMT"]
prime_date_cols = ["CREATION_DATE","LAST_STAEMENT_DATE","LAST_PAYMENT_DATE","DOB","CLOSURE_DATE"]

# ========================= 1. Load Data ==========================
prime_files = glob.glob("prime/*.csv")
prime_df_list = []

print("Loading CSV files...")
for file in prime_files:
    temp_df = pd.read_csv(
        file, 
        encoding='latin', 
        dtype={col: "string" for col in prime_string_cols + prime_int_cols + prime_float_cols},
        parse_dates=prime_date_cols
    ).rename(columns={'RIM_NO': 'RIMNO', "NAME": "PRODUCT_NAME"})
    
    prime_df_list.append(temp_df)
    
prime_df = pd.concat(prime_df_list, ignore_index=True)

# ========================= 2. Casting & Reporting ==========================
print("\n--- Casting Columns and Checking Nulls ---")


print("\n-> String Columns:")
apply_cast_and_report(prime_df, prime_string_cols, 'string')

print("\n-> Float Columns:")
apply_cast_and_report(prime_df, prime_float_cols, 'float')

print("\n-> Integer Columns:")
apply_cast_and_report(prime_df, prime_int_cols, 'int')

print("\n-> Date Columns:")
apply_cast_and_report(prime_df, prime_date_cols, 'date')

# ========================= 3. Cleanup & Final Info ==========================
prime_columns_to_drop = ["MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT", "TOTAL_HOLD" ,"ORGANIZATION","JOINING_FEE","ANNUAL_FEE","LAST_PAYMENT_AMOUNT", "LAST_PAYMENT_DATE", "SETTLEMENT AMT"]
existing_cols_to_drop = [col for col in prime_columns_to_drop if col in prime_df.columns]
prime_df = prime_df.drop(columns=existing_cols_to_drop)

print("\n--- Final Dataframe Info ---")
print(prime_df.info())

# ========================= 4. Load Transaction Data ==========================
print("\nLoading Transaction files...")

transaction_string_cols = ["DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES", "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME"]
transaction_int_cols = ["RIMNO","CCY","MCC","SETTLEMENT CCY"]
transaction_float_cols = ["ORIG AMOUNT","EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]
transaction_date_cols = ["TRXN DATE", "POST DATE"]

transaction_files = glob.glob("transaction/*.csv")
transaction_df_list = []

for file in transaction_files:
    temp_df = pd.read_csv(
        file,
        encoding='latin',
        dtype={col: "string" for col in transaction_string_cols + transaction_int_cols + transaction_float_cols},
        parse_dates=transaction_date_cols
    )
    transaction_df_list.append(temp_df)

transaction_df = pd.concat(transaction_df_list, ignore_index=True)


# ========================= 5. Transaction Casting & Reporting ==========================
print("\n--- Casting Transaction Columns and Checking Nulls ---")
print("\n-> Transaction String Columns:")
apply_cast_and_report(transaction_df, transaction_string_cols, 'string')
print("\n-> Transaction Float Columns:")
apply_cast_and_report(transaction_df, transaction_float_cols, 'float')
print("\n-> Transaction Integer Columns:")
apply_cast_and_report(transaction_df, transaction_int_cols, 'int')
print("\n-> Transaction Date Columns:")
apply_cast_and_report(transaction_df, transaction_date_cols, 'date')


transaction_columns_to_drop = ["POST DATE","ORIG AMOUNT","EMBEDDED _FEE", "SETTLEMENT AMT", "SETTLEMENT CCY", "SOURCES"]
existing_cols_to_drop = [col for col in transaction_columns_to_drop if col in transaction_df.columns]
transaction_df = transaction_df.drop(columns=existing_cols_to_drop)

# ========================= 6. Final Transaction Info ==========================
print("\n--- Final Transaction Dataframe Info ---")
print(transaction_df.info())


critical_columns = ['RIMNO']
prime_df = drop_empty_records(prime_df, critical_columns)

prime_df["GENDER"] = prime_df["GENDER"].fillna("Unknown")
transaction_df['ORIG AMOUNT'] = transaction_df['ORIG AMOUNT'].fillna(transaction_df['ORIG AMOUNT'].median())

prime_df['DOB_IS_MISSING'] = prime_df['DOB'].isna().astype(int) # 1 if missing, 0 if present
if not prime_df['DOB'].dropna().empty:
    median_dob_int = prime_df['DOB'].dropna().astype('int64').median()
    median_dob = pd.to_datetime(median_dob_int)
    prime_df['DOB'] = prime_df['DOB'].fillna(median_dob)

prime_df['LAST_STAEMENT_IS_MISSING'] = prime_df['LAST_STAEMENT_DATE'].isna().astype(int)
latest_txns = transaction_df.groupby("RIMNO")["TRXN DATE"].max()
prime_df['LAST_STAEMENT_DATE'] = prime_df['LAST_STAEMENT_DATE'].fillna(prime_df['RIMNO'].map(latest_txns))
prime_df['LAST_STAEMENT_DATE'] = prime_df['LAST_STAEMENT_DATE'].fillna(prime_df['CREATION_DATE'])


transaction_df["MERCHNAME"] = transaction_df["MERCHNAME"].fillna("UNKNOWN")
transaction_df["BANKBRANCH"] = transaction_df["BANKBRANCH"].fillna("UNKNOWN")
transaction_df["TRXN COUNTRY"] = transaction_df["TRXN COUNTRY"].fillna("UNKNOWN")

extraction_date = pd.to_datetime("today")
prime_df["ACCOUNT_TENURE_MONTHS"] = (extraction_date - prime_df["CREATION_DATE"]).dt.days /30
prime_df["DAYS_SINCE_LAST_PAYMENT"] = (extraction_date - prime_df["LAST_PAYMENT_DATE"]).dt.days
prime_df["TIME_TO_CHURN_DAYS"]= (prime_df['CLOSURE_DATE'] - prime_df['CREATION_DATE']).dt.days
print(prime_df.head())
print(transaction_df.head())

txn_recency = transaction_df.groupby("RIMNO")["TRXN DATE"].max().reset_index()
txn_recency["DAYS_SINCE_LAST_TXN"] = (extraction_date - txn_recency["TRXN DATE"]).dt.days

prime_df = prime_df.merge(txn_recency[["RIMNO","DAYS_SINCE_LAST_TXN"]], on="RIMNO", how="left") 

prime_df["CARD_REPLACEMENT_FREQ"] =prime_df[['FIRST_REPLACED_CARD', 'SECOND_REPLACED_CARD','THIRD_REPLACED_CARD']].sum(axis=1)

#=========================transaction Encoding ==========================
trxn_categorical_cols = ["DESCRIPTION","CCY","MCC","SETTLEMENT CCY","BANKBRANCH","TRXN COUNTRY","REVERSAL FLAG"]
for col in trxn_categorical_cols:
    unique_values = transaction_df[col].unique()
    print(f"Unique values in {col}: {unique_values}")

#=====================Prime Encoding=========================
prime_categorical_cols = ["BRANCH_NAME",'FIRST_REPLACED_CARD', 'SECOND_REPLACED_CARD','THIRD_REPLACED_CARD']
for col in prime_categorical_cols:
    unique_values = prime_df[col].unique()
    print(f"Unique values in {col}: {unique_values}")


#========================= Transaction & Prime Encoding Export ==========================
trxn_categorical_cols = ["DESCRIPTION","CCY","MCC","SETTLEMENT CCY","BANKBRANCH","TRXN COUNTRY","REVERSAL FLAG"]
prime_categorical_cols = ["BRANCH_NAME",'FIRST_REPLACED_CARD', 'SECOND_REPLACED_CARD','THIRD_REPLACED_CARD']

with open("category_frequencies.txt", "w", encoding="utf-8") as file:
    
    file.write("========================= Transaction Frequencies =========================\n")
    for col in trxn_categorical_cols:
        file.write(f"\n--- Frequency of unique values in {col} ---\n")
        file.write(transaction_df[col].value_counts(dropna=False).to_string() + "\n")

    file.write("\n========================= Prime Frequencies =========================\n")
    for col in prime_categorical_cols:
        file.write(f"\n--- Frequency of unique values in {col} ---\n")
        file.write(prime_df[col].value_counts(dropna=False).to_string() + "\n")

print("Frequencies successfully exported to 'category_frequencies.txt'")



Product_frequency_threshold = 500

# Prime
prime_counts = prime_df["PRODUCT_NAME"].value_counts()
prime_df["PRODUCT_NAME"] = prime_df["PRODUCT_NAME"].map(
    lambda x: x if prime_counts.get(x, 0) >= Product_frequency_threshold else "another product"
)

# Transaction
transaction_counts = transaction_df["Description"].value_counts()
transaction_df["PRODUCT_NAME"] = transaction_df["PRODUCT_NAME"].map(
    lambda x: x if transaction_counts.get(x, 0) >= Product_frequency_threshold else "another product"
)

MCC_frequency_threshold = 5000
transaction_counts = transaction_df["MCC"].value_counts()
transaction_df["MCC"] = transaction_df["MCC"].map(
    lambda x: x if transaction_counts.get(x, 0) >= MCC_frequency_threshold else "another product"
)


user_item_matrix = pd.crosstab(prime_df['RIMNO'], prime_df['PRODUCT_NAME'])
user_item_matrix.columns = [f"HAS_PROD_{col.strip().replace(' ', '_')}" for col in user_item_matrix.columns]
prime_df = prime_df.drop_duplicates(subset=['RIMNO']).merge(user_item_matrix, on='RIMNO', how='left')

# 1. Standard RFM (Recency, Frequency, Monetary)
rfm_features = transaction_df.groupby('RIMNO').agg(
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
    index='RIMNO', 
    columns='MCC', 
    aggfunc='sum', 
    fill_value=0
)
# Rename columns for clarity (e.g., MCC_5411_SPEND)
mcc_spend.columns = [f"MCC_{str(col)}_SPEND" for col in mcc_spend.columns]
mcc_spend = mcc_spend.reset_index()

transaction_df['IS_FOREIGN_TRXN'] = (transaction_df['TRXN COUNTRY'] != 'EG').fillna(False).astype(int)

travel_features = transaction_df.groupby('RIMNO').agg(
    FOREIGN_TRXN_COUNT=('IS_FOREIGN_TRXN', 'sum'),
    FOREIGN_SPEND_RATIO=('IS_FOREIGN_TRXN', 'mean') # Percentage of transactions done abroad
).reset_index()

# Merge all transactional features back to prime_df
prime_df = prime_df.merge(rfm_features, on='RIMNO', how='left')
prime_df = prime_df.merge(mcc_spend, on='RIMNO', how='left')
prime_df = prime_df.merge(travel_features, on='RIMNO', how='left')

# Fill NaNs for customers with no transactions
trxn_feature_cols = rfm_features.columns.tolist() + mcc_spend.columns.tolist() + travel_features.columns.tolist()
trxn_feature_cols.remove('RIMNO')
prime_df[trxn_feature_cols] = transaction_df[trxn_feature_cols].fillna(0)


# 1. Calculate Age
prime_df['AGE'] = (extraction_date - prime_df['DOB']).dt.days // 365

# 2. Create Age Bands (Categorical feature for easier targeting)
bins = [18, 25, 35, 50, 65, 100]
labels = ['18-25', '26-35', '36-50', '51-65', '65+']
prime_df['AGE_GROUP'] = pd.cut(prime_df['AGE'], bins=bins, labels=labels, right=True)

# 3. Credit Limit Bands
# Helps group high-net-worth vs standard customers
limit_bins = [0, 10000, 50000, 100000, 500000, np.inf]
limit_labels = ['Low', 'Medium', 'High', 'Very High', 'Premium']
prime_df['LIMIT_BAND'] = pd.cut(prime_df['CREDIT_LIMIT'], bins=limit_bins, labels=limit_labels)

cols_to_ohe = ['GENDER', 'AGE_GROUP', 'LIMIT_BAND', 'CUSTOMER_TYPE', 'ACTIVATED']
prime_df = pd.get_dummies(prime_df, columns=cols_to_ohe, drop_first=True)


# ========================= Data Scaling ==========================
print("\n--- Applying Data Scaling ---")
float_cols_to_scale = prime_df.select_dtypes(include=['float64', 'float32']).columns.tolist()

# 2. Initialize the Scaler
# StandardScaler transforms data to have a mean of 0 and standard deviation of 1.
scaler = StandardScaler()

if float_cols_to_scale:
    print(f"Scaling the following {len(float_cols_to_scale)} float columns:")
    for col in float_cols_to_scale:
        print(f" - {col}")
        
    prime_df[float_cols_to_scale] = scaler.fit_transform(prime_df[float_cols_to_scale])
    
    print("\nScaling complete. Statistical summary of a few scaled columns:")
    print(prime_df[float_cols_to_scale[:5]].describe().round(3))
else:
    print("No float columns found to scale.")
    

# ========================= Final NaN Handling ==========================
print("\n--- Final Check for NaN Values ---")

cols_with_nans = prime_df.columns[prime_df.isna().any()].tolist()

if not cols_with_nans:
    print("Clean sweep! No NaN values found in the final DataFrame.")
else:
    print(f"Found {len(cols_with_nans)} columns with missing values. Filling with median...")
    
    # 2. Iterate through columns with NaNs
    for col in cols_with_nans:
        # Check if the column is numerical (int or float)
        if pd.api.types.is_numeric_dtype(prime_df[col]):
            median_val = prime_df[col].median()
            prime_df[col] = prime_df[col].fillna(median_val)
            print(f" - Filled [{col}] with median: {median_val:.4f}")
        else:
            # For non-numeric columns remaining, fill with 'Unknown' or mode
            mode_val = prime_df[col].mode()[0] if not prime_df[col].mode().empty else "Unknown"
            prime_df[col] = prime_df[col].fillna(mode_val)
            print(f" - Filled non-numeric [{col}] with mode/placeholder: {mode_val}")

print("\nFinal NaN check complete. Total NaNs in DF:", prime_df.isna().sum().sum())
