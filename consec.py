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

rimnos_list = []
# (Optional sanity check print order)
print("\n=== Correct File Order ===")
for f in files:
    print(os.path.basename(f))


print("\n=== Loading RIMNOs ===")
for f in files:
    rimnos_list.append(extract_rimnos(f))

# ================= MONTH TO MONTH COMPARISON =================
for i in range(1, len(files)):
    
    prev_file = files[i - 1]
    curr_file = files[i]
    
    prev_rimnos = rimnos_list[i - 1]
    curr_rimnos = rimnos_list[i]
    
    prev_label = f"{prev_file[-12:-4]}"
    curr_label = f"{curr_file[-12:-4]}"
    
    missing = prev_rimnos - curr_rimnos
    new = curr_rimnos - prev_rimnos
    
    output_file = f"RIMNO_{prev_label}_TO_{curr_label}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        
        f.write(f"================ {prev_label} → {curr_label} ================\n\n")
        
        f.write(f"Previous RIMNOs: {len(prev_rimnos)}\n")
        f.write(f"Current RIMNOs: {len(curr_rimnos)}\n\n")
        
        f.write("----------- NEW (new customers) -----------\n")
        f.write(f"Count: {len(new)}\n")
        f.write(str(new) + "\n")

        f.write("----------- MISSING (churn candidates) -----------\n")
        f.write(f"Count: {len(missing)}\n")
        f.write(str(missing) + "\n\n")

        for j in range(len(rimnos_list)):
            if j == i - 1 or j == i:
                continue

            intersection = missing & rimnos_list[j]
            if len(intersection) != 0:
                f.write(f"In {files[j][-12:-4]}: {len(intersection)} of the missing RIMNOs exist\n")
                f.write(str(intersection) + "\n\n")


print("\nDone: Fully chronological RIMNO audit generated.")
