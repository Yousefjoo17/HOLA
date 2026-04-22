
import pandas as pd

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


def drop_empty_records(df, columns):
    """
    Drops rows from a dataframe if they are missing data in the specified columns.
    Prints a summary of how many rows were removed.
    """
    # Safeguard: if a single string is passed instead of a list, convert it
    if isinstance(columns, str):
        columns = [columns]
        
    # Filter the list to only include columns that actually exist in the dataframe
    valid_cols = [col for col in columns if col in df.columns]
    
    if not valid_cols:
        print(f"\nWarning: None of the specified columns {columns} exist in the dataframe. No rows dropped.")
        return df
        
    # Count rows before dropping
    rows_before = len(df)
    
    # Drop the empty records
    cleaned_df = df.dropna(subset=valid_cols, how='any')
    
    # Report the results
    rows_after = len(cleaned_df)
    rows_dropped = rows_before - rows_after
    
    print(f"\n--- Dropping Empty Records ---")
    print(f"Checked columns: {valid_cols}")
    print(f"Dropped {rows_dropped} rows.")
    print(f"Total rows remaining: {rows_after}")
    
    return cleaned_df


def check_missing_values(df):
    """
    Analyzes a DataFrame for missing values and returns a summary table 
    containing only the columns that have missing data.
    """
    # 1. Count missing values per column
    missing_counts = df.isna().sum()
    
    # 2. Filter out columns that have 0 missing values
    missing_counts = missing_counts[missing_counts > 0]
    
    # 3. Calculate the percentage of missing data
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100
    
    # 4. Create a clean summary DataFrame
    summary_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Percentage (%)': missing_percentages.round(2)
    })
    
    # 5. Sort from highest number of missing values to lowest
    summary_df = summary_df.sort_values(by='Missing Count', ascending=False)
    
    # Check if the dataframe had no missing values at all
    if summary_df.empty:
        print("Great news! There are no missing values in this DataFrame.")
        return None
        
    return summary_df

