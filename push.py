import pandas as pd 
import numpy as np
from datetime import datetime
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
import glob

prime_string_cols =["BRANCH_NAME","ACTIVATED","STATUS","STATUS_NAME","NAME","GENDER","CUSTOMER_TYPE","Card account status ","ORGANIZATION"]
prime_int_cols = ["RIMNO"]
prime_float_cols = ["AVAILABLE_LIMIT","LEDGER_BALANCE","LAST_PAYMENT_AMOUNT","OVERDUEAMOUNT"]
prime_date_cols =["DOB","CREATION_DATE","LAST_STAEMENT_DATE","LAST_PAYMENT_DATE","CLOSURE_DATE"]

def parse_int(x):
    if pd.isna(x):
        return pd.NA
    x_str = str(x).strip()
    x_clean  = x_str.replace(",", "")
    if x_clean.endswith(".00"):     
        x_clean = x_clean[:-3]
    try:
        return int(x_clean)
    except ValueError:
        return np.nan 
def parse_float(x):
    if pd.isna(x):
        return pd.nan
    x_str = str(x).strip()
    x_clean  = x_str.replace(",", "")
    try:
        return float(x_clean)
    except ValueError:
        return np.nan
    
#=========================prime==========================
prime_files =glob.glob("prime/*.csv")
prime_df_list = []
for file in prime_files:
    df = pd.read_csv(file)
    prime_df_list.append(pd.read_csv(file,encoding='latin',dtype={col:"string" for col in prime_string_cols+prime_int_cols+prime_float_cols},
                                      parse_dates=prime_date_cols).rename(columns={'RIM_NO:':'RIMNO'}))
    
prime_df = pd.concat(prime_df_list, ignore_index=True)

for col in prime_float_cols:
    prime_df[col] = prime_df[col].apply(parse_float)
for col in prime_int_cols:
    prime_df[col] = prime_df[col].apply(parse_int)

prime_df = prime_df.drop(columns=["MAPPING_ACCNO","MIN_PAYMENT","OVER_LIMIT","TOTAL_HOLD"])
print(prime_df.info())

#=========================transaction==========================
transacion_string_cols = ["DESCRIPTION","MERCHNAME","MERCH ID","SOURCES","BANKBRANCH","TRXN COUNTRY","REVERSAL FLAG"]
transacion_int_cols = []
transacion_float_cols = []
transacion_date_cols = ["TRXN DATE","POST DATE"]
transaction_files = glob.glob("transaction/*.csv")\

transaction_df_list = []
for file in transaction_files:
    transaction_df_list.append(pd.read_excel(file,encoding='latin',dtype={col:"string" for col in transacion_string_cols+transacion_int_cols+transacion_float_cols},
                                      parse_dates=transacion_date_cols))
transaction_df = pd.concat(transaction_df_list, ignore_index=True)

print(transaction_df.info())

prime_df["GENDER"] = prime_df[col].fillna("Unknown")
transaction_df['ORIG AMOUNT'] = transaction_df['ORIG AMOUNT'].fillna(transaction_df['ORIG AMOUNT'].median())
prime_df['ANNUAL_FEE'] =prime_df['ANNUAL_FEE'].fillna(prime_df['ANNUAL_FEE'].median())
outlier_cols = ["BILLING AMT","SETTLEMENT AMT","ORIG AMOUNT"]
for col in outlier_cols:
    if col in transaction_df:
        p99= transaction_df[col].quantile(0.99)
        transaction_df[col] = np.where(transaction_df[col]>p99, p99, transaction_df[col])
p99_overdue = prime_df["OVERDUEAMOUNT"].quantile(0.99)
prime_df["OVERDUEAMOUNT"] = np.where(prime_df["OVERDUEAMOUNT"]>p99_overdue, p99_overdue, prime_df["OVERDUEAMOUNT"])

extraction_date = pd.to_datetime("today")
prime_df["ACCOUNT_TENURE_MONTHS"] = (extraction_date - prime_df["CREATION_DATE"]).dt.days /30
prime_df["DAYS_SINCE_LAST_PAYMENT"] = (extraction_date - prime_df["LAST_PAYMENT_DATE"]).dt.days
prime_df["TIME_TO_CHURN_DAYS"]= (prime_df['CLOSURE_DATE'] - prime_df['CREATION_DATE']).dt.days
print(prime_df.head())
print(transaction_df.head())

txn_recency = transaction_df.groupby("RIMNO")["TRXN DATE"].max().reset_index()
txn_recency["DAYS_SINCE_LAST_TXN"] = (extraction_date - txn_recency["TRXN DATE"]).dt.days

prime_df = prime_df.merge(txn_recency[["RIMNO","DAYS_SINCE_LAST_TXN"]], on="RIMNO", how="left") 

prime_df['UTILIZATION_Ratio']= prime_df['LEDGER_BALANCE'] / prime_df['CREDIT_LIMIT'].replace({0: np.nan})
prime_df["AVAILABLE_CREDIT_RATIO"] = prime_df["AVAILABLE_LIMIT"] / prime_df["CREDIT_LIMIT"].replace({0: np.nan})

prime_df["FEE_TO_LIMIT_RATIO"] = (prime_df["JOINING_FEE"] + prime_df["ANNUAL_FEE"]) / prime_df["CREDIT_LIMIT"].replace({0: np.nan})
prime_df["CARD_REPLACEMENT_FREQ"] =prime_df[['FIRST_REPLACED_CARD', 'SECOND_REPLACED_CARD','THIRD_REPLACED_CARD']].sum(axis=1)

prime_df.to_csv("cleaned_prime.csv", index=False)

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

# Open a text file in write mode ('w')
with open("category_frequencies.txt", "w", encoding="utf-8") as file:
    
    # Write Transaction data
    file.write("========================= Transaction Frequencies =========================\n")
    for col in trxn_categorical_cols:
        file.write(f"\n--- Frequency of unique values in {col} ---\n")
        # Convert the Series to a string and write it, followed by a newline character
        file.write(transaction_df[col].value_counts(dropna=False).to_string() + "\n")

    # Write Prime data
    file.write("\n========================= Prime Frequencies =========================\n")
    for col in prime_categorical_cols:
        file.write(f"\n--- Frequency of unique values in {col} ---\n")
        file.write(prime_df[col].value_counts(dropna=False).to_string() + "\n")

print("Frequencies successfully exported to 'category_frequencies.txt'")



threshold = 1000

# Prime
prime_counts = prime_df["PRODUCT_NAME"].value_counts()
prime_df["PRODUCT_NAME"] = prime_df["PRODUCT_NAME"].map(
    lambda x: x if prime_counts.get(x, 0) >= threshold else "another product"
)

# Transaction
transaction_counts = transaction_df["PRODUCT_NAME"].value_counts()
transaction_df["PRODUCT_NAME"] = transaction_df["PRODUCT_NAME"].map(
    lambda x: x if transaction_counts.get(x, 0) >= threshold else "another product"
)
