import glob
import pandas as pd

prime_files = sorted(glob.glob("prime/*.csv"))

previous_rimnos = None

for i, file in enumerate(prime_files):
    print(f"\n================ FILE {i+1}: {file} ================\n")
    
    df = pd.read_csv(file, encoding='latin')
    
    # normalize column name if needed
    if "RIM_NO" in df.columns:
        df = df.rename(columns={"RIM_NO": "RIMNO"})
    
    current_rimnos = set(df["RIMNO"].dropna().unique())
    
    print(f"Total RIMNOs in file: {len(current_rimnos)}")
    
    if previous_rimnos is not None:
        
        # 1. Missing from previous month (expected but not found now)
        missing_from_current = previous_rimnos - current_rimnos
        
        # 2. New RIMNOs appearing this month
        new_rimnos = current_rimnos - previous_rimnos
        
        print(f"\n--- Missing RIMNOs from previous file ---")
        print(f"Count: {len(missing_from_current)}")
        if len(missing_from_current) > 0:
            print(missing_from_current)
        
        print(f"\n--- New RIMNOs in current file ---")
        print(f"Count: {len(new_rimnos)}")
        if len(new_rimnos) > 0:
            print(new_rimnos)
    
    else:
        print("First file baseline established.")
    
    previous_rimnos = current_rimnos