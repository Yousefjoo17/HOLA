import glob
import pandas as pd
import os
import re
from datetime import datetime

month_map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def get_month_year(file_path):
    """
    Extracts proper datetime from filename like:
    cleaned_JUL_2025.csv
    """
    name = os.path.basename(file_path).upper()
    match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_(\d{4})", name)
    
    if match:
        month = month_map[match.group(1)]
        year = int(match.group(2))
        return datetime(year, month, 1)
    
    # fallback very early date
    return datetime(1900, 1, 1)


def extract_rimnos(file):
    df = pd.read_csv(file, encoding='latin')
    
    if "RIM_NO" in df.columns:
        df = df.rename(columns={"RIM_NO": "RIMNO"})
    
    return set(df["RIMNO"].dropna().unique())


# ================= SORT CORRECTLY =================
files = glob.glob("prime/*.csv")
files = sorted(files, key=get_month_year)

# (Optional sanity check print order)
print("\n=== Correct File Order ===")
for f in files:
    print(os.path.basename(f))


# ================= MONTH TO MONTH COMPARISON =================
for i in range(1, len(files)):
    
    prev_file = files[i - 1]
    curr_file = files[i]
    
    prev_rimnos = extract_rimnos(prev_file)
    curr_rimnos = extract_rimnos(curr_file)
    
    prev_label = os.path.basename(prev_file).replace(".csv", "")
    curr_label = os.path.basename(curr_file).replace(".csv", "")
    
    missing = prev_rimnos - curr_rimnos
    new = curr_rimnos - prev_rimnos
    
    output_file = f"RIMNO_{prev_label}_TO_{curr_label}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        
        f.write(f"================ {prev_label} → {curr_label} ================\n\n")
        
        f.write(f"Previous RIMNOs: {len(prev_rimnos)}\n")
        f.write(f"Current RIMNOs: {len(curr_rimnos)}\n\n")
        
        f.write("----------- MISSING (churn candidates) -----------\n")
        f.write(f"Count: {len(missing)}\n")
        f.write(str(missing) + "\n\n")
        
        f.write("----------- NEW (new customers) -----------\n")
        f.write(f"Count: {len(new)}\n")
        f.write(str(new) + "\n")

print("\nDone: Fully chronological RIMNO audit generated.")
