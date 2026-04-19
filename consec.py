import glob
import pandas as pd
import os

prime_files = sorted(glob.glob("prime/*.csv"))

def extract_rimnos(file):
    df = pd.read_csv(file, encoding='latin')
    if "RIM_NO" in df.columns:
        df = df.rename(columns={"RIM_NO": "RIMNO"})
    return set(df["RIMNO"].dropna().unique())

for i in range(1, len(prime_files)):
    
    prev_file = prime_files[i - 1]
    curr_file = prime_files[i]
    
    prev_rimnos = extract_rimnos(prev_file)
    curr_rimnos = extract_rimnos(curr_file)
    
    # Try to create readable month label from filenames
    prev_name = os.path.splitext(os.path.basename(prev_file))[0]
    curr_name = os.path.splitext(os.path.basename(curr_file))[0]
    
    period_name = f"{prev_name}_TO_{curr_name}"
    output_file = f"RIMNO_{period_name}.txt"
    
    missing_from_current = prev_rimnos - curr_rimnos
    new_rimnos = curr_rimnos - prev_rimnos
    
    with open(output_file, "w", encoding="utf-8") as f:
        
        f.write(f"================ {prev_name} → {curr_name} ================\n\n")
        
        f.write(f"Previous file RIMNO count: {len(prev_rimnos)}\n")
        f.write(f"Current file RIMNO count: {len(curr_rimnos)}\n\n")
        
        f.write("----------- Missing from current (potential churn) -----------\n")
        f.write(f"Count: {len(missing_from_current)}\n")
        f.write(str(missing_from_current) + "\n\n")
        
        f.write("----------- New RIMNOs (new customers) -----------\n")
        f.write(f"Count: {len(new_rimnos)}\n")
        f.write(str(new_rimnos) + "\n")

print("All month-to-month RIMNO reports exported.")
